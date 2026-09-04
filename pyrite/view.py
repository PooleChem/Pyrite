from __future__ import annotations
from typing import TYPE_CHECKING
from wsgiref.validate import bad_header_value_re

from IPython.display import display, HTML
from ipywidgets import IntSlider, interactive, widgets, VBox,  Layout


import html as _html
import py3Dmol
from numpy.typing import NDArray
from rdkit import Chem

if TYPE_CHECKING:
    from pyrite import Mol
from pyrite.atom_consts import AtomType


class Viewer:
    """Visualisation class.

    Allows for the easy visualization of pyrite objects.

    Parameters
    ----------
    *args : Mol, Bounds
        All arguments are considered as objects to show.
    width : int, default 400
        The width of the viewer.
    height : int, default 400
        The height of the viewer.

    """

    _HOVER_LABEL_IDX_JS_CALLBACK = """function(atom,viewer,event,container) {
                   if(!atom.label) {
                    atom.label = viewer.addLabel(atom.index,{position: atom, backgroundColor: 'mintcream', fontColor:'black'});
                   }}"""
    _HOVER_LABEL_RES_JS_CALLBACK = """function(atom,viewer,event,container) {
                   if(!atom.label) {
                    atom.label = viewer.addLabel(atom.resn+atom.resi,{position: atom, backgroundColor: 'mintcream', fontColor:'black'});
                   }}"""

    _UNHOVER_LABEL_JS_CALLBACK = """function(atom,viewer) {
                                       if(atom.label) {
                                        viewer.removeLabel(atom.label);
                                        delete atom.label;
                                       }
                                    }"""

    def __init__(self, *args, width: int = 400, height: int = 400, options=None):
        self._width = width
        self._height = height
        self._iframe = None
        self._widget = None
        self._slider = None
        self.view = py3Dmol.view(
            width=width, height=height, options={"doAssembly": True}
        )
        self.max_m_id = -1
        self.add(*args, options=options)

        self._vs = []
        self._v_draw_options = {}
        self._v_m_id = -1
        self._interactive = None
        self.__interactive_first = None
        self._ligand = None
        self._out = None
        self._view_state_key = f"pyrite_view_state_{id(self)}"



    def add(self, *args, options=None):
        """Adds all positional arguments to the viewer.

        This method adds all arguments to the viewer, optionally with draw options `options`.

        Currently supports:

        * :class:`~pyrite.Mol`
        * :class:`~pyrite.bounds.Bounds`

        Parameters
        ----------
        *args : Mol, Bounds
            The objects to add.
        options : dictionary
            The draw options to apply.

        Returns
        -------
        Viewer

        """
        for arg in args:
            if not callable(getattr(arg, "_viewer_add_", None)):
                raise ValueError(
                    f"Argument {arg} does not have a _viewer_add_ method, and"
                    f"can thus not be automatically added to this Viewer."
                )

        for arg in args:
            new_m_id = arg._viewer_add_(self, self.max_m_id, options)  # noqa
            self.max_m_id = new_m_id

        return self

    def _set_ligand(self, v_id):
        self.view.removeModel(self._v_m_id)
        vn = self._vs[v_id]

        conf_id = self._ligand.update(vn, new_conf=True)
        mblock_n = Chem.MolToMolBlock(self._ligand, confId=conf_id)
        self._ligand.RemoveConformer(conf_id)
        self.view.addModel(mblock_n, "mol")
        self.view.setStyle(
            {"model": self._v_m_id},
            {
                "stick": {
                    "colorscheme": self._v_draw_options["colorscheme"],
                    "hidden": False,
                }
            },
        )
        if "note" in self._v_draw_options:
            self._set_hover(self._ligand, self.max_m_id, self._v_draw_options)
        # Re-render the iframe so the new pose is visible across frontends
        self._render_iframe()
        # self.view.update()

    def add_v(self, mol: Mol, v: NDArray, slider: bool = True, options=None):
        """Adds a molecule with poses, and an optional pose selection slider, to the viewer.

        This method adds a molecule, with poses `v`, to the viewer,
        optionally with draw options `options`. Optionally, a `slider` can be added, which allows
        for the selection of the displayed pose. If `slider` is ``False``, all poses are shown.

        Parameters
        ----------
        mol : Mol
            The objects to add.
        v : numpy.ndarray
            The poses to show.
        slider : bool, default True
            Whether the pose selection slider is shown. If `slider` is ``False``,
            all poses are shown.
        options : dictionary
            The draw options to apply.

        Returns
        -------
        Viewer

        """

        if options is None:
            options = {}

        self._ligand = mol
        self._vs = v
        self._v_draw_options = mol.draw_options.copy()
        self._v_draw_options.update(options)

        if slider:
            # Update to first v
            vn = self._vs[0]

            conf_id = self._ligand.update(vn, new_conf=True)
            mblock = Chem.MolToMolBlock(self._ligand, confId=conf_id)
            self._ligand.RemoveConformer(conf_id)

            self.view.addModel(mblock, "mol")
            self._v_m_id = self.max_m_id + 1
            self.max_m_id = self._v_m_id

            self.view.setStyle(
                {"model": self._v_m_id},
                {
                    "stick": {
                        "colorscheme": self._v_draw_options["colorscheme"],
                        "hidden": False,
                    }
                },
            )

            # self.__interactive_first = True
            # self._interactive = interactive(
            #     self._set_ligand,
            #     v_id=IntSlider(
            #         min=0,
            #         max=len(v) - 1,
            #         step=1,
            #         continuous_update=True,
            #         description="Pose:",
            #     ),
            # )
            self._slider = IntSlider(
                min=0,
                max=len(v) - 1,
                step=1,
                continuous_update=True,
                description="Pose:",
            )
            self._slider.observe(lambda ch: self._set_ligand(ch["new"]), names="value")

        else:
            for var in self._vs:
                conf_id = mol.update(var, new_conf=True)
                mblock = Chem.MolToMolBlock(mol, confId=conf_id)
                mol.RemoveConformer(conf_id)
                self.view.addModel(mblock, "mol")
                self.max_m_id += 1
                self.view.setStyle(
                    {"model": self.max_m_id},
                    {
                        "stick": {
                            "colorscheme": self._v_draw_options["colorscheme"],
                        }
                    },
                )

        if "note" in self._v_draw_options:
            self._set_hover(self._ligand, self.max_m_id, self._v_draw_options)

        return self

    def _set_hover(self, obj, model_id, options=None):
        if "note" not in options:
            options["note"] = ""
        match options["note"].lower():
            case "idx":
                hover_js_callback = Viewer._HOVER_LABEL_IDX_JS_CALLBACK
            case "type":
                hover_js_callback = (
                    """function(atom,viewer,event,container) {
                        let atom_types = """
                    + str([AtomType(t).name for t in obj.atom_types])
                    + """
                          if(!atom.label) {
                              atom.label = viewer.addLabel(atom_types[atom.index],{position: atom, backgroundColor: 'mintcream', fontColor:'black'});
                          }}"""
                )
            case "res":
                hover_js_callback = Viewer._HOVER_LABEL_RES_JS_CALLBACK
            case _:
                hover_js_callback = None

        if hover_js_callback is not None:
            self.view.setHoverable(
                {"model": model_id},
                True,
                hover_js_callback,
                Viewer._UNHOVER_LABEL_JS_CALLBACK,
            )

    def show(self):
        """Return a displayable widget (works in Jupyter, VSCode, and PyCharm)."""
        return self.as_widget()

    def as_widget(self):
        # Build once, then reuse so slider callbacks update the same viewer instance
        if self._widget is None:
            self._render_iframe()
            if self._slider is None:
                self._widget = self._iframe
            else:
                self._widget = VBox([self._iframe, self._slider])
        return self._widget

    def _ipython_display_(self):
        display(self.as_widget())


    def _render_iframe(self):
        """Render the current py3Dmol view into an iframe and persist camera state across reloads."""
        if self._iframe is None:
            self._iframe = widgets.HTML()

        # Generate base HTML
        page = self.view.write_html(fullpage=True)

        # Inject JS to persist/restore camera
        key = self._view_state_key
        persist_js = f"""
    <script>
    (function() {{
      const KEY = {key!r};

      function findViewer() {{
        // py3Dmol often uses 'viewer', but we search just in case
        if (window.viewer && typeof window.viewer.getView === "function") return window.viewer;
        for (const k of Object.keys(window)) {{
          const v = window[k];
          if (v && typeof v.getView === "function" && typeof v.setView === "function" && typeof v.zoomTo === "function") {{
            return v;
          }}
        }}
        return null;
      }}

      function restoreOrZoom(viewer) {{
        try {{
          const saved = localStorage.getItem(KEY);
          if (saved) {{
            viewer.setView(JSON.parse(saved));
            viewer.render();
            return;
          }}
        }} catch (e) {{}}
        // No saved view -> zoom to content once
        viewer.zoomTo();
        viewer.render();
      }}

      function startSaving(viewer) {{
        // Save periodically
        setInterval(() => {{
          try {{
            localStorage.setItem(KEY, JSON.stringify(viewer.getView()));
          }} catch (e) {{}}
        }}, 250);
      }}

      // Wait until py3Dmol has created the viewer
      const t = setInterval(() => {{
        const viewer = findViewer();
        if (!viewer) return;
        clearInterval(t);
        restoreOrZoom(viewer);
        startSaving(viewer);
      }}, 50);
    }})();
    </script>
    """

   
        # Put our script right before </body> if possible
        if "</body>" in page:
            page = page.replace("</body>", persist_js + "\n</body>")
        else:
            page = page + persist_js

        srcdoc = _html.escape(page, quote=True)
        self._iframe.value = (
            f'<iframe srcdoc="{srcdoc}" '
            f'width="{self._width}" height="{self._height}" '
            f'style="border:0;"></iframe>'
        )