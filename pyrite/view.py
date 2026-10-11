from __future__ import annotations

import html as _html
import json
import re
from typing import TYPE_CHECKING

import py3Dmol
from IPython.display import HTML
from numpy.typing import NDArray
from rdkit import Chem

if TYPE_CHECKING:
    from pyrite import Mol
from pyrite.atom_consts import AtomType


class Viewer:
    """Visualisation class.

    Allows for the easy visualization of pyrite objects in a Jupyter notebook (also in VSCode
    and PyCharm), using py3Dmol.

    Parameters
    ----------
    *args : Mol, Bounds
        All arguments are considered as objects to show.
    width : int, default 400
        The width of the viewer, in pixels.
    height : int, default 400
        The height of the viewer, in pixels.
    options : dict, optional
        The draw options for all objects in `args`, see :meth:`add`.

    See Also
    --------
    pyrite.Mol.viewer : A viewer of one molecule.
    pyrite.Mol.set_draw_options : How a molecule is drawn.

    Examples
    --------
    >>> Viewer(receptor, ligand, pocket).show()
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
        self.view = py3Dmol.view(width=width, height=height, options={"doAssembly": True})
        self.max_m_id = -1
        # The model with the poses of add_v, its number of poses, and its style (reapplied on every
        # pose: a style is set on the atoms of the current frame only).
        self._slider = None
        # The models of every molecule added, to zoom to: (molecule, model ids).
        self._models = []
        self._zoom = None
        self.add(*args, options=options)

    def add(self, *args, options=None):
        """Add all positional arguments to the viewer.

        This method adds all arguments to the viewer, optionally with draw options `options`.

        Currently supports:

        * :class:`~pyrite.Mol`
        * :class:`~pyrite.bounds.Bounds`

        Parameters
        ----------
        *args : Mol, Bounds
            The objects to add.
        options : dict, optional
            The draw options to apply, on top of the object's own ``draw_options``. The key
            ``"note"`` adds a label when hovering over an atom: ``"idx"`` (the atom index),
            ``"type"`` (the Pyrite atom type) or ``"res"`` (the residue).

        Returns
        -------
        Viewer
            This viewer, so calls can be chained.

        Raises
        ------
        ValueError
            If an argument cannot be shown (it has no ``_viewer_add_`` method).

        See Also
        --------
        add_v : Add a molecule with poses.

        Examples
        --------
        >>> viewer = Viewer(receptor)
        >>> viewer.add(ligand, pocket, options={"note": "idx"}).show()
        """
        for arg in args:
            if not callable(getattr(arg, "_viewer_add_", None)):
                raise ValueError(
                    f"Argument {arg} does not have a _viewer_add_ method, and"
                    f"can thus not be automatically added to this Viewer."
                )

        for arg in args:
            new_m_id = arg._viewer_add_(self, self.max_m_id, options)  # noqa
            if new_m_id != self.max_m_id:
                self._models.append((arg, list(range(self.max_m_id + 1, new_m_id + 1))))
            self.max_m_id = new_m_id

        return self

    def add_v(self, mol: Mol, v: NDArray, slider: bool = True, options=None):
        """Add a molecule with poses, and an optional pose selection slider, to the viewer.

        This method adds a molecule, with poses `v`, to the viewer, optionally with draw options
        `options`. Optionally, a `slider` can be added, which allows for the selection of the
        displayed pose. If `slider` is ``False``, all poses are shown.

        Parameters
        ----------
        mol : Mol
            The molecule the poses belong to.
        v : Poses or numpy.ndarray
            The poses to show, in the layout of `mol`.
        slider : bool, default True
            Whether the pose selection slider is shown. If `slider` is ``False``, all poses are
            shown at once.
        options : dict, optional
            The draw options to apply, see :meth:`add`.

        Returns
        -------
        Viewer
            This viewer, so calls can be chained.

        See Also
        --------
        add : Add objects.

        Examples
        --------
        >>> Viewer(receptor).add_v(ligand, docked_poses).show()
        """
        if options is None:
            options = {}
        draw_options = mol.draw_options.copy()
        draw_options.update(options)
        style = {"stick": {"colorscheme": draw_options["colorscheme"]}}

        blocks = [Chem.MolToMolBlock(mol.to_rdkit(pose)) for pose in v]
        first = self.max_m_id + 1
        if slider:
            # All poses in one model, as frames: the slider switches frames in the browser.
            self.view.addModelsAsFrames("$$$$\n".join(blocks) + "$$$$\n", "sdf")
            self.max_m_id += 1
            self.view.setStyle({"model": self.max_m_id}, style)
            self._slider = (self.max_m_id, len(blocks), style)
        else:
            for block in blocks:
                self.view.addModel(block, "mol")
                self.max_m_id += 1
                self.view.setStyle({"model": self.max_m_id}, style)

        if "note" in draw_options:
            self._set_hover(mol, self.max_m_id, draw_options)

        # The poses are what a viewer with poses is about: zoom to them, unless told otherwise.
        self._models.append((mol, list(range(first, self.max_m_id + 1))))
        if self._zoom is None:
            self._zoom = self._models[-1][1]
        return self

    def zoom_to(self, *objects):
        """Zoom to some of the molecules in the viewer.

        By default, a viewer shows everything in it, except when it has poses (see :meth:`add_v`):
        then it zooms to the poses. ``zoom_to()`` without arguments shows everything again.

        Parameters
        ----------
        *objects : Mol
            The molecules to zoom to, as added with :meth:`add` or :meth:`add_v`. A molecule added
            with both is zoomed to with all of its models.

        Returns
        -------
        Viewer
            This viewer, so calls can be chained.

        Raises
        ------
        ValueError
            If an object is not in the viewer as a molecule.

        See Also
        --------
        add : Add objects.

        Examples
        --------
        >>> Viewer(receptor, ligand).zoom_to(ligand)
        """
        ids = []
        for obj in objects:
            found = [m for o, models in self._models if o is obj for m in models]
            if not found:
                raise ValueError(f"{obj!r} is not a molecule in this viewer.")
            ids += found
        self._zoom = ids
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
        """Return the viewer, to display in a notebook.

        The viewer is plain HTML and JavaScript, the pose slider included: it works in Jupyter,
        VS Code and PyCharm, and in a notebook exported to HTML or a documentation page, without a
        running kernel. In a notebook, the viewer also shows itself when it is the last expression
        of a cell.

        Returns
        -------
        IPython.display.HTML
            The viewer.

        See Also
        --------
        as_widget : The viewer as a widget, to place in a layout.

        Examples
        --------
        >>> Viewer(receptor, ligand).show()
        """
        return HTML(self._repr_html_())

    def as_widget(self):
        """Return the viewer as an ipywidgets widget, to place in a layout of your own.

        The widget holds the same HTML as :meth:`show`, so the pose slider works without a kernel
        too. Requires ipywidgets.

        Returns
        -------
        ipywidgets.HTML
            The viewer.

        See Also
        --------
        show : Show the viewer.

        Examples
        --------
        >>> import ipywidgets
        >>> ipywidgets.HBox([Viewer(ligand).as_widget(), Viewer(receptor).as_widget()])
        """
        import ipywidgets

        return ipywidgets.HTML(self._repr_html_())

    def _repr_html_(self):
        page = self.view.write_html(fullpage=True)
        name = re.search(r"var (viewer_\w+) = null", page).group(1)

        height = self._height
        controls = ""
        zoom = json.dumps(self._zoom or [])
        script = (
            # Model indices, not models: 3Dmol copies a selection, and a model refers to the viewer.
            f"const zoom = {zoom};"
            f"{name}.zoomTo(zoom.length ? {{model: zoom}} : {{}}); {name}.render();"
        )
        if self._slider is not None:
            model, n, style = self._slider
            height += 32
            controls = (
                '<div style="display:flex;align-items:center;gap:8px;height:32px;'
                'font:13px system-ui,sans-serif;color:#444">'
                f'<input id="pose" type="range" min="0" max="{n - 1}" value="0" style="flex:1">'
                f'<span id="label" style="padding-right:8px;white-space:nowrap">Pose 1 / {n}</span></div>'
            )
            script += f"""
  const model = {name}.getModel({model}), style = {json.dumps(style)};
  document.getElementById("pose").addEventListener("input", (event) => {{
    const i = Number(event.target.value);
    document.getElementById("label").textContent = `Pose ${{i + 1}} / {n}`;
    Promise.resolve(model.setFrame(i)).then(() => {{
      model.setStyle({{}}, style);
      {name}.render();
    }});
  }});"""

        # Zoom once all models are in (py3Dmol zooms before adding them), then wire the slider.
        page = (
            '<!doctype html><html><body style="margin:0">'
            f"{page}{controls}<script>$3Dmolpromise.then(() => {{ {script} }});</script>"
            "</body></html>"
        )
        return (
            f'<iframe srcdoc="{_html.escape(page, quote=True)}" width="{self._width}" '
            f'height="{height}" style="border:0;max-width:100%"></iframe>'
        )
