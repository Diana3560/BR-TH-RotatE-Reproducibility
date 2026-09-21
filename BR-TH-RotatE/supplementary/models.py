from __future__ import annotations
import torch
from throtate_repro.model import THRotatEInteraction, build_pykeen_model_class
from throtate_repro.multidataset_experiment import _model_spec
from throtate_repro.magnitude_preserving_normalization import magnitude_preserving_constants, build_bounded_adaptive_magnitude_preserving_model_class
from supplementary.per_relation_model import build_per_relation_control_bundle

class PlainMPInteraction(THRotatEInteraction):
    """Exact D0/D1 parameterization, adding only two fixed scale buffers."""
    def __init__(self, *, c_h, c_r, **kwargs):
        super().__init__(**kwargs)
        constants = magnitude_preserving_constants(c_h, c_r)
        self.register_buffer('scale_h', torch.tensor(constants['transh_multiplier']))
        self.register_buffer('scale_r', torch.tensor(constants['rotate_multiplier']))
    def forward(self, h, r, t):
        dh, dr = self.component_distances(h, r, t)
        a, b = self.fusion_weights.unbind()
        return -(a*self.scale_h*dh+b*self.scale_r*dr)

class PlainMP(build_pykeen_model_class()):
    def __init__(self, *, c_h, c_r, **kwargs):
        super().__init__(**kwargs)
        old = self.interaction
        self.interaction = PlainMPInteraction(c_h=c_h,c_r=c_r,transh_norm=old.transh_norm,
            transh_power_norm=old.transh_power_norm,rotate_norm=old.rotate_norm)
        with torch.no_grad(): self.interaction.raw_fusion_weights.copy_(old.raw_fusion_weights)

def spec_for(cfg, context, name, hp, scales=None):
    if name.startswith(('RatE','CompoundE','PairRE')):
        from stage3.models import build_rate_model_class, build_compounde_model_class
        from pykeen.models import PairRE
        family = name.replace('_recip','')
        cls = {'RatE':build_rate_model_class,'CompoundE':build_compounde_model_class,'PairRE':lambda:PairRE}[family]()
        return {'model':cls,'model_kwargs':{'embedding_dim':hp['embedding_dim']},
                'reciprocal':name.endswith('_recip'),'implementation':cls.__module__+'.'+cls.__name__}

    # Supplementary control: directly learn one bounded scalar angle offset for every
    # modeled forward/inverse relation state.  This deliberately removes all BR
    # Train-only structural statistics and frequency shrinkage while keeping the
    # D2 geometry, reciprocal supervision, global anchor, Delta, loss and budget.
    if name == 'D2_PerRelation':
        spec = _model_spec(cfg, context, 'D2', hp['delta'])
        bundle = build_per_relation_control_bundle(
            relation_to_id=context['full'].relation_to_id,
            modeled_relations=context['modeled_relations'],
            num_internal_relations=int(context['reciprocal_training'].num_relations),
        )
        spec['model_kwargs']['relation_feature_tensor'] = bundle.features
        spec['model_kwargs']['relation_reliability_tensor'] = bundle.reliability
        spec['implementation'] = (
            'throtate_repro.bounded_adaptive_v26.BoundedAdaptiveTHRotatE'
            '+reciprocal+direct_per_relation_bounded_angle'
        )
        spec['control_design'] = {
            'name': 'D2_PerRelation',
            'formula': 'theta_rd = theta_global + Delta * tanh(delta_rd)',
            'active_relation_direction_states': len(bundle.rows),
            'direct_angle_parameters': bundle.trainable_angle_parameters,
            'uses_relation_identity_only': True,
            'uses_train_structural_statistics': False,
            'uses_frequency_shrinkage': False,
            'uses_business_role_features': False,
            'reciprocal_training': True,
            'zero_initialized_offsets': True,
            'one_hot_assignment': bundle.rows,
        }
        return spec

    core = name.split('_')[0]
    spec = _model_spec(cfg, context, core, hp['delta'])
    spec['model_kwargs']['embedding_dim'] = hp['embedding_dim']
    if name == 'D2_noShrinkage':
        q = spec['model_kwargs']['relation_reliability_tensor']
        spec['model_kwargs']['relation_reliability_tensor'] = torch.ones_like(q)
    if name in ['D2_constRole','D2_constNoRole']:
        x = spec['model_kwargs']['relation_feature_tensor'].clone()
        if name == 'D2_constNoRole':
            names = context['reciprocal_bundle'].feature_names
            x[:,[names.index('role_diagnostic'),names.index('role_procedural')]] = 0.
        # Both models gain the same one-column intercept feature (11 router parameters).
        spec['model_kwargs']['relation_feature_tensor'] = torch.cat([x,torch.ones_like(x[:,:1])],dim=1)
    if name.endswith('_MPNorm'):
        if scales is None: raise ValueError('MPNorm requires frozen Train-only calibration')
        if core in ['D0','D1']:
            spec['model'] = PlainMP
            spec['model_kwargs'].update(c_h=scales['c_h'],c_r=scales['c_r'])
        else:
            spec['model'] = build_bounded_adaptive_magnitude_preserving_model_class()
            spec['model_kwargs'].update(transh_mean=scales['c_h'],rotate_mean=scales['c_r'])
    spec['implementation'] = spec['model'].__module__+'.'+spec['model'].__name__
    return spec
