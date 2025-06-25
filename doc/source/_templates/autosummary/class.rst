.. raw:: html

   <div class="prename">{{ module }}.</div>
   <div class="empty"></div>

{{ name }}
{{ underline }}

.. currentmodule:: {{ module }}

{% block autoclass %}
.. autoclass:: {{ objname }}
   :private-members: _score
   :no-members:
   :no-inherited-members:
   :no-special-members:
{% endblock %}

  {% block methods %}
   .. HACK -- the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
      .. autosummary::
         :toctree:
      {% for item in all_methods %}
         {%- if (not item.startswith('_')) or (item in ['_score']) %}
         {{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
      {% for item in inherited_members %}
         {%- if item in ['_score'] %}
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
         {%- if not item.startswith('_') %}
         {{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
  {% endif %}
  {% endblock %}
