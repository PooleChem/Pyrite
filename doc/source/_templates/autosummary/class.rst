{{ name }}
{{ underline }}

.. currentmodule:: {{ module }}

{#- The private methods a subclass implements, documented on the base class that defines them only:
    a concrete scoring function's page shows its public API. The guide page "Writing a scoring
    function" explains them. -#}
{%- set hook_map = {
    "ScoringFunction": ["_score", "_batch_scores", "_score_and_gradient"],
    "_KNNScoringFunction": ["_kernel", "_mask", "_score_field"],
} %}
{%- set hooks = hook_map.get(name, []) %}
{#- What AtomType (an IntEnum) inherits from int: not Pyrite's, and with int's own docstrings. -#}
{%- set builtin_members = ['from_bytes', 'to_bytes', 'as_integer_ratio', 'bit_count', 'bit_length', 'conjugate', 'is_integer', 'denominator', 'numerator', 'real', 'imag'] %}

{% block autoclass %}
.. autoclass:: {{ objname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:
{% endblock %}

{% if hooks %}
.. rubric:: Methods for subclasses

.. autosummary::
{% for item in hooks %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}

  {% block methods %}
   .. HACK -- the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
      .. autosummary::
         :toctree:
      {% for item in all_methods %}
         {%- if ((not item.startswith('_')) or (item in hooks) or item == '__call__') and item not in builtin_members %}
         {{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
      {% for item in inherited_members %}
         {%- if item in hooks %}
         {{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
  {% endblock %}

  {% block attributes %}
  {% if attributes %}
   .. HACK -- the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
      .. autosummary::
         :toctree:
      {% for item in all_attributes %}
         {%- if not item.startswith('_') and item not in builtin_members %}
         {{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
  {% endif %}
  {% endblock %}
