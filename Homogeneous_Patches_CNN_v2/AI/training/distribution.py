"""
distribution.py
Hierarchical patch balancing weights.
"""
from collections import defaultdict
from torch.utils.data import WeightedRandomSampler

def get_hierarchical_weights(records, patches_per_img=200):
    # records: [(path, brand, model, device)]
    
    # Counts
    brand_models = defaultdict(set)
    model_devices = defaultdict(set)
    device_images = defaultdict(int)
    
    for _, b, m, d in records:
        brand_models[b].add(m)
        model_devices[m].add(d)
        device_images[d] += 1
        
    num_brands = len(brand_models)
    
    weights = []
    for _, b, m, d in records:
        n_models = len(brand_models[b])
        n_devices = len(model_devices[m])
        n_images = device_images[d]
        
        # Paper Eq 4 logic: W = 1 / (n_b * n_m * n_d * n_i)
        w = 1.0 / (num_brands * n_models * n_devices * n_images)
        # Each image has patches_per_img patches
        weights.extend([w] * patches_per_img)
        
    return weights

def get_sampler(records, patches_per_img=200):
    weights = get_hierarchical_weights(records, patches_per_img)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler
