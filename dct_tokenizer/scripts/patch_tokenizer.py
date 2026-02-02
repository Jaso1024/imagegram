#!/usr/bin/env python3
"""
Hierarchical Patch Tokenizer for n-gram image generation.

Key idea: Tokens should be spatially coherent for n-grams to capture structure.

Approach:
1. Divide image into patches (e.g., 8x8, 16x16, 32x32)
2. Each patch gets a token based on its content
3. Raster scan order: left-to-right, top-to-bottom
4. Coarse-to-fine: larger patches first, then smaller patches for detail

This gives us:
- Spatial locality: adjacent tokens = adjacent patches
- Coarse-to-fine: early tokens = global structure, later = details
- Simple quantization: use color/intensity histograms or vector quantization

Token ordering:
[32x32 patches in raster order] [16x16 patches] [8x8 patches]
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from sklearn.cluster import MiniBatchKMeans
import pickle
import time

class PatchTokenizer:
    """Hierarchical patch tokenizer with learned codebook."""
    
    PRESETS = {
        # (image_size, patch_sizes, codebook_sizes)
        # More patches at coarser levels for global structure
        'tiny':   (256, [32], [256]),           # 64 patches * 1 level = 64 tokens
        'small':  (256, [32, 16], [256, 256]),  # 64 + 256 = 320 tokens
        'medium': (256, [32, 16, 8], [256, 256, 256]),  # 64 + 256 + 1024 = 1344 tokens
        'large':  (256, [16, 8], [512, 512]),   # 256 + 1024 = 1280 tokens
    }
    
    def __init__(self, preset='small', codebooks=None):
        cfg = self.PRESETS[preset]
        self.image_size = cfg[0]
        self.patch_sizes = cfg[1]
        self.codebook_sizes = cfg[2]
        self.preset = preset
        
        # Calculate tokens per level
        self.tokens_per_level = []
        for ps in self.patch_sizes:
            n_patches = (self.image_size // ps) ** 2
            self.tokens_per_level.append(n_patches)
        
        self.tokens_per_image = sum(self.tokens_per_level)
        self.vocab_size = max(self.codebook_sizes)
        
        # Codebooks for each level (learned via k-means)
        self.codebooks = codebooks  # List of (kmeans_model, patch_size)
        
        print(f"PatchTokenizer: {preset}")
        print(f"  Patch sizes: {self.patch_sizes}")
        print(f"  Tokens per level: {self.tokens_per_level}")
        print(f"  Total tokens: {self.tokens_per_image}")
        print(f"  Vocab size: {self.vocab_size}")
    
    def extract_patches(self, image: np.ndarray, patch_size: int) -> np.ndarray:
        """Extract patches from image in raster order."""
        h, w = image.shape[:2]
        patches = []
        
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                patch = image[y:y+patch_size, x:x+patch_size]
                patches.append(patch.flatten())
        
        return np.array(patches, dtype=np.float32)
    
    def reconstruct_from_patches(self, patches: np.ndarray, patch_size: int) -> np.ndarray:
        """Reconstruct image from patches."""
        n_patches_side = self.image_size // patch_size
        n_channels = patches.shape[1] // (patch_size * patch_size)
        
        image = np.zeros((self.image_size, self.image_size, n_channels), dtype=np.float32)
        
        idx = 0
        for y in range(n_patches_side):
            for x in range(n_patches_side):
                patch = patches[idx].reshape(patch_size, patch_size, n_channels)
                image[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size] = patch
                idx += 1
        
        return image
    
    def fit(self, images: list, n_samples_per_level: int = 100000):
        """Learn codebooks from sample images."""
        print(f"Fitting codebooks on {len(images)} images...")
        
        self.codebooks = []
        
        for level, (patch_size, codebook_size) in enumerate(zip(self.patch_sizes, self.codebook_sizes)):
            print(f"  Level {level}: patch_size={patch_size}, codebook_size={codebook_size}")
            
            # Collect patches from all images
            all_patches = []
            for img in images:
                if img.shape[:2] != (self.image_size, self.image_size):
                    img_pil = Image.fromarray(img)
                    img_pil = img_pil.resize((self.image_size, self.image_size), Image.LANCZOS)
                    img = np.array(img_pil)
                
                patches = self.extract_patches(img, patch_size)
                all_patches.append(patches)
            
            all_patches = np.vstack(all_patches)
            
            # Subsample if too many
            if len(all_patches) > n_samples_per_level:
                idx = np.random.choice(len(all_patches), n_samples_per_level, replace=False)
                all_patches = all_patches[idx]
            
            print(f"    Training on {len(all_patches)} patches...")
            
            # Train k-means
            kmeans = MiniBatchKMeans(
                n_clusters=codebook_size,
                batch_size=1000,
                n_init=3,
                random_state=42
            )
            kmeans.fit(all_patches)
            
            self.codebooks.append((kmeans, patch_size))
            print(f"    Done. Inertia: {kmeans.inertia_:.2f}")
        
        return self
    
    def save_codebooks(self, path: str):
        """Save learned codebooks."""
        with open(path, 'wb') as f:
            pickle.dump({
                'preset': self.preset,
                'codebooks': [(km.cluster_centers_, ps) for km, ps in self.codebooks]
            }, f)
        print(f"Saved codebooks to {path}")
    
    def load_codebooks(self, path: str):
        """Load learned codebooks."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.codebooks = []
        for centers, patch_size in data['codebooks']:
            kmeans = MiniBatchKMeans(n_clusters=len(centers))
            kmeans.cluster_centers_ = centers
            kmeans._n_features_out = centers.shape[1]
            self.codebooks.append((kmeans, patch_size))
        
        print(f"Loaded codebooks from {path}")
        return self
    
    def tokenize(self, image: np.ndarray) -> np.ndarray:
        """Tokenize an image."""
        if self.codebooks is None:
            raise ValueError("Codebooks not initialized. Call fit() or load_codebooks() first.")
        
        if image.shape[:2] != (self.image_size, self.image_size):
            img_pil = Image.fromarray(image)
            img_pil = img_pil.resize((self.image_size, self.image_size), Image.LANCZOS)
            image = np.array(img_pil)
        
        tokens = []
        
        for kmeans, patch_size in self.codebooks:
            patches = self.extract_patches(image, patch_size)
            level_tokens = kmeans.predict(patches)
            tokens.extend(level_tokens)
        
        return np.array(tokens, dtype=np.uint16)
    
    def detokenize(self, tokens: np.ndarray) -> np.ndarray:
        """Detokenize tokens back to image."""
        if self.codebooks is None:
            raise ValueError("Codebooks not initialized.")
        
        # Start with coarsest level, then blend finer levels
        images = []
        
        idx = 0
        for level, (kmeans, patch_size) in enumerate(self.codebooks):
            n_tokens = self.tokens_per_level[level]
            level_tokens = tokens[idx:idx + n_tokens]
            idx += n_tokens
            
            # Get patch centroids
            patches = kmeans.cluster_centers_[level_tokens]
            
            # Reconstruct
            img = self.reconstruct_from_patches(patches, patch_size)
            images.append(img)
        
        # Simple blending: start with coarsest, add residuals from finer
        # For now, just use the finest level
        result = images[-1]
        
        return np.clip(result, 0, 255).astype(np.uint8)


def create_simple_codebook(preset='small'):
    """Create a simple color-based codebook without learning."""
    tokenizer = PatchTokenizer(preset)
    
    # Create simple codebooks based on color quantization
    codebooks = []
    
    for patch_size, codebook_size in zip(tokenizer.patch_sizes, tokenizer.codebook_sizes):
        # Create a grid of colors in RGB space
        n_per_dim = int(np.cbrt(codebook_size)) + 1
        
        colors = []
        for r in np.linspace(0, 255, n_per_dim):
            for g in np.linspace(0, 255, n_per_dim):
                for b in np.linspace(0, 255, n_per_dim):
                    # Create a uniform patch of this color
                    patch = np.full((patch_size * patch_size * 3,), [r, g, b] * (patch_size * patch_size // 3 + 1)[:patch_size * patch_size * 3])
                    colors.append(np.array([r, g, b] * (patch_size * patch_size), dtype=np.float32))
        
        colors = np.array(colors[:codebook_size])
        
        # Create fake kmeans
        kmeans = MiniBatchKMeans(n_clusters=codebook_size)
        kmeans.cluster_centers_ = colors
        kmeans._n_features_out = colors.shape[1]
        
        codebooks.append((kmeans, patch_size))
    
    tokenizer.codebooks = codebooks
    return tokenizer


def test_simple():
    """Test with simple color codebook."""
    print("Creating simple tokenizer...")
    tokenizer = create_simple_codebook('small')
    
    # Test on gradient image
    x = np.linspace(0, 255, 256)
    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
    test_img[:, :, 0] = x[np.newaxis, :]
    test_img[:, :, 1] = x[:, np.newaxis]
    test_img[:, :, 2] = 128
    
    print("\nTokenizing...")
    tokens = tokenizer.tokenize(test_img)
    print(f"Tokens: {len(tokens)}, range [{tokens.min()}, {tokens.max()}]")
    print(f"First 20: {tokens[:20]}")
    
    print("\nDetokenizing...")
    recon = tokenizer.detokenize(tokens)
    
    mse = np.mean((test_img.astype(float) - recon.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2 / (mse + 1e-10))
    print(f"PSNR: {psnr:.2f} dB")


def train_on_images(image_dir: str, output_path: str, preset: str, max_images: int = 5000):
    """Train codebooks on images from a directory."""
    tokenizer = PatchTokenizer(preset)
    
    # Load images
    print(f"Loading images from {image_dir}...")
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG']:
        image_paths.extend(Path(image_dir).rglob(ext))
    
    if len(image_paths) > max_images:
        image_paths = np.random.choice(image_paths, max_images, replace=False)
    
    images = []
    for p in image_paths:
        try:
            img = Image.open(p).convert('RGB')
            img = img.resize((tokenizer.image_size, tokenizer.image_size), Image.LANCZOS)
            images.append(np.array(img))
        except:
            pass
    
    print(f"Loaded {len(images)} images")
    
    # Train
    tokenizer.fit(images)
    tokenizer.save_codebooks(output_path)
    
    return tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train', help='Directory of images to train on')
    parser.add_argument('--output', default='codebooks.pkl')
    parser.add_argument('--preset', default='small')
    args = parser.parse_args()
    
    if args.test:
        test_simple()
    elif args.train:
        train_on_images(args.train, args.output, args.preset)
    else:
        test_simple()
