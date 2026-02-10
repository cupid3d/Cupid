import importlib

__attributes = {
    'SparseStructure': 'sparse_structure',
    'SparseUVStructure': 'sparse_uv_structure',
    'ImageConditionedSparseUVStructure': 'sparse_uv_structure',
    
    'SparseFeat2Render': 'sparse_feat2render',
    'SLat2Render':'structured_latent2render',
    'Slat2RenderGeo':'structured_latent2render',
    
    'SparseStructureLatent': 'sparse_structure_latent',
    'TextConditionedSparseStructureLatent': 'sparse_structure_latent',
    'ImageConditionedSparseStructureLatent': 'sparse_structure_latent',
    'SparseStructureWithLatent': 'sparse_structure_latent',
    'ImageConditionedSparseStructureWithLatent': 'sparse_structure_latent',

    'SparseUVStructureLatent': 'sparse_uv_structure_latent',
    'ImageConditionedSparseUVStructureLatent': 'sparse_uv_structure_latent',
    'SparseUVStructureWithLatent': 'sparse_uv_structure_latent',
    'ImageConditionedSparseUVStructureWithLatent': 'sparse_uv_structure_latent',
    
    'SLat': 'structured_latent',
    'TextConditionedSLat': 'structured_latent',
    'ImageConditionedSLat': 'structured_latent',
    'SlatWithUVTransforms': 'structured_latent',
    'TextConditionedSLatWithUVTransforms': 'structured_latent',
    'ImageConditionedSLatWithUVTransforms': 'structured_latent',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
    from .sparse_structure import SparseStructure
    from .sparse_uv_structure import (
        SparseUVStructure, 
        ImageConditionedSparseUVStructure,
    )
    
    from .sparse_feat2render import SparseFeat2Render
    from .structured_latent2render import (
        SLat2Render,
        Slat2RenderGeo,
    )
    
    from .sparse_structure_latent import (
        SparseStructureLatent,
        TextConditionedSparseStructureLatent,
        ImageConditionedSparseStructureLatent,
        SparseStructureWithLatent,
        ImageConditionedSparseStructureWithLatent,
    )
    
    from .sparse_uv_structure_latent import (
        SparseUVStructureLatent,
        ImageConditionedSparseUVStructureLatent,
        SparseUVStructureWithLatent,
        ImageConditionedSparseUVStructureWithLatent,
    )
    
    from .structured_latent import (
        SLat,
        TextConditionedSLat,
        ImageConditionedSLat,
        SlatWithUVTransforms,
        ImageConditionedSLatWithUVTransforms,
        TextConditionedSLatWithUVTransforms,
    )
    