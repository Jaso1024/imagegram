#!/usr/bin/env python3
"""Multi-shard query wrapper for Fastgram indices.

Fastgram's GramEngine already supports multiple shards by passing a list of
index directories. However, for very large shard counts (or for distributed
querying), it can be useful to *fan out* queries and merge results.

This module implements a MultiShardGramEngine that:
- Loads shards in groups (each group is a GramEngine over multiple shard dirs)
- Fans out queries across groups (thread pool)
- Merges results with correct Infini-gram suffix fallback semantics

Supported operations:
- count(prompt_ids)
- ntd(prompt_ids, max_support)

Notes on correctness for ntd:
Each group returns a suffix_len and prompt_cnt for the *longest* suffix that
has support *within that group*. Globally, we must use the maximum suffix_len
that has support in any group, and ignore groups that backed off to shorter
suffixes (since their count at the longer suffix is 0).

Example usage:
  engine = MultiShardGramEngine(
      index_root='/root/cc12m_indices',
      eos_token_id=65535,
      vocab_size=256,
      group_size=10,
      threads_per_engine=1,
  )
  print(engine.ntd([128,128]))
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union


@dataclass
class _NtdGroupResult:
    prompt_cnt: int
    approx: bool
    suffix_len: int
    cont_cnt_by_tok: Dict[int, int]


class MultiShardGramEngine:
    def __init__(
        self,
        index_root: Union[str, Path],
        eos_token_id: int,
        vocab_size: int,
        version: int = 4,
        token_dtype: str = "u16",
        group_size: int = 10,
        threads_per_engine: int = 1,
        max_support: int = 1000,
        fanout_threads: Optional[int] = None,
        shard_glob: str = "[0-9][0-9][0-9][0-9][0-9]",
    ) -> None:
        from fastgram import GramEngine

        self.index_root = Path(index_root)
        self.eos_token_id = int(eos_token_id)
        self.vocab_size = int(vocab_size)
        self.version = int(version)
        self.token_dtype = token_dtype
        self.group_size = int(group_size)
        self.threads_per_engine = int(threads_per_engine)
        self.max_support = int(max_support)

        # Enumerate shard dirs
        shard_dirs = []
        for p in sorted(self.index_root.glob(shard_glob)):
            if (p / "table.0").exists() and (p / "offset.0").exists() and (p / "tokenized.0").exists():
                shard_dirs.append(str(p))
        if not shard_dirs:
            raise ValueError(f"No shard index dirs found under {self.index_root}")

        self.shard_dirs: List[str] = shard_dirs

        # Group shard dirs
        self.groups: List[List[str]] = [
            shard_dirs[i : i + self.group_size] for i in range(0, len(shard_dirs), self.group_size)
        ]

        # Build group engines
        self.engines: List[GramEngine] = []
        for g in self.groups:
            self.engines.append(
                GramEngine(
                    index_dir=g,
                    eos_token_id=self.eos_token_id,
                    vocab_size=self.vocab_size,
                    version=self.version,
                    token_dtype=self.token_dtype,
                    threads=self.threads_per_engine,
                    max_support=self.max_support,
                )
            )

        if fanout_threads is None:
            fanout_threads = min(len(self.engines), max(1, os.cpu_count() or 1))
        self.fanout_threads = int(fanout_threads)

    def _merge_count(self, results: List[dict]) -> dict:
        total = 0
        approx = False
        for r in results:
            if "error" in r:
                continue
            total += int(r.get("count", 0))
            approx = approx or bool(r.get("approx", False))
        return {"count": total, "approx": approx}

    def count(self, input_ids: List[int]) -> dict:
        with ThreadPoolExecutor(max_workers=self.fanout_threads) as ex:
            futs = [ex.submit(e.count, input_ids) for e in self.engines]
            results = [f.result() for f in futs]
        return self._merge_count(results)

    def _ntd_one(self, engine, prompt_ids: List[int], max_support: int) -> _NtdGroupResult:
        r = engine.ntd(prompt_ids, max_support=max_support)
        if "error" in r:
            return _NtdGroupResult(prompt_cnt=0, approx=True, suffix_len=0, cont_cnt_by_tok={})

        prompt_cnt = int(r.get("prompt_cnt", 0) or 0)
        approx = bool(r.get("approx", False))
        suffix_len = int(r.get("suffix_len", 0) or 0)

        # result_by_token_id maps tok -> {cont_cnt, prob}
        cont_cnt_by_tok: Dict[int, int] = {}
        for tok, d in (r.get("result_by_token_id") or {}).items():
            cont_cnt_by_tok[int(tok)] = int(d.get("cont_cnt", 0))

        return _NtdGroupResult(
            prompt_cnt=prompt_cnt,
            approx=approx,
            suffix_len=suffix_len,
            cont_cnt_by_tok=cont_cnt_by_tok,
        )

    def ntd(self, prompt_ids: List[int], max_support: Optional[int] = None) -> dict:
        if max_support is None:
            max_support = self.max_support

        # Fan out across groups
        results: List[_NtdGroupResult] = []
        with ThreadPoolExecutor(max_workers=self.fanout_threads) as ex:
            futs = [ex.submit(self._ntd_one, e, prompt_ids, max_support) for e in self.engines]
            for f in as_completed(futs):
                results.append(f.result())

        # Determine global suffix_len (Infini-gram semantics)
        global_suffix = 0
        for r in results:
            if r.prompt_cnt > 0 and r.suffix_len > global_suffix:
                global_suffix = r.suffix_len

        # Only keep groups that used the global suffix
        rel = [r for r in results if r.suffix_len == global_suffix and r.prompt_cnt > 0]

        total_prompt = sum(r.prompt_cnt for r in rel)
        approx = any(r.approx for r in results)

        merged_counts: Dict[int, int] = {}
        for r in rel:
            for tok, cnt in r.cont_cnt_by_tok.items():
                merged_counts[tok] = merged_counts.get(tok, 0) + cnt

        # Trim to max_support by cont_cnt
        items = sorted(merged_counts.items(), key=lambda kv: kv[1], reverse=True)
        items = items[:max_support]

        total = total_prompt if total_prompt > 0 else 1
        result_by_token_id = {int(tok): {"cont_cnt": int(cnt), "prob": float(cnt) / total} for tok, cnt in items}

        return {
            "prompt_cnt": int(total_prompt),
            "result_by_token_id": result_by_token_id,
            "approx": bool(approx),
            "suffix_len": int(global_suffix),
        }


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", required=True)
    ap.add_argument("--eos", type=int, default=65535)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--group-size", type=int, default=10)
    ap.add_argument("--fanout-threads", type=int, default=0)
    ap.add_argument("--mode", choices=["count", "ntd"], default="ntd")
    ap.add_argument("--prompt", type=str, default="128,128")
    ap.add_argument("--max-support", type=int, default=50)
    args = ap.parse_args()

    prompt_ids = [int(x) for x in args.prompt.split(",") if x.strip()]
    fanout_threads = args.fanout_threads if args.fanout_threads > 0 else None

    engine = MultiShardGramEngine(
        index_root=args.index_root,
        eos_token_id=args.eos,
        vocab_size=args.vocab,
        group_size=args.group_size,
        fanout_threads=fanout_threads,
        max_support=max(args.max_support, 1),
    )

    if args.mode == "count":
        print(engine.count(prompt_ids))
    else:
        out = engine.ntd(prompt_ids, max_support=args.max_support)
        # print top 10
        items = sorted(out.get("result_by_token_id", {}).items(), key=lambda kv: -kv[1]["prob"])[:10]
        print({k: out[k] for k in ["prompt_cnt", "approx", "suffix_len"]})
        for tok, d in items:
            print(tok, d)


if __name__ == "__main__":
    _cli()
