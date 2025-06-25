
{% extends "autosummary/class.rst" %}

{% block autoclass %}
.. autoclass:: {{ fullname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:
   :exclude-members: ClearComputedProps, ClearProp, Compute2DCoords, ComputeGasteigerCharges, Debug, GetAromaticAtoms, GetAtomWithIdx, GetAtomsMatchingQuery, GetBondBetweenAtoms, GetBondWithIdx, GetBonds, GetBoolProp, GetConformers, GetDoubleProp, GetIntProp, GetNumAtoms, GetNumBonds, GetNumConformers, GetNumHeavyAtoms, GetProp, GetPropNames, GetPropsAsDict, GetRingInfo, GetStereoGroups, GetSubstructMatch, GetSubstructMatches, GetUnsignedProp, HasProp, HasQuery, HasSubstructMatch, NeedsUpdatePropertyCache, RemoveAllConformers, SetBoolProp, SetDoubleProp, SetIntProp, SetProp, SetUnsignedProp, ToBinary, UpdatePropertyCache
{% endblock %}

{% block methods %}
.. HACK -- the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
  .. autosummary::
     :toctree:
  {% for item in all_methods %}
     {%- if (not item.startswith('_') or item in ['__call__', '__mul__', '__getitem__', '__len__', '__pow__']) and ((item not in inherited_members) or (item in ['AddConformer', 'RemoveConformer', 'GetAtoms', 'GetConformer'])) %}
     {{ name }}.{{ item }}
     {%- endif -%}
  {%- endfor %}
  {% for item in inherited_members %}
     {%- if item in ['__call__', '__mul__', '__getitem__', '__len__', '__pow__'] %}
     {{ name }}.{{ item }}
     {%- endif -%}
  {%- endfor %}
{% endblock %}


