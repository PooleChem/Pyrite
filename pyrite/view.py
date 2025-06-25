from __future__ import annotations
from typing import TYPE_CHECKING
from IPython.display import display
from ipywidgets import IntSlider, interactive

import py3Dmol
from numpy.typing import NDArray
from rdkit import Chem

if TYPE_CHECKING:
    from pyrite import Ligand
from pyrite.atom_consts import AtomType


class Viewer:
    """Visualisation class.

    Allows for the easy visualization of pyrite objects.

    Parameters
    ----------
    *args : Ligand, Receptor, Bounds
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

    def add(self, *args, options=None):
        """Adds all positional arguments to the viewer.

        This method adds all arguments to the viewer, optionally with draw options `options`.

        Currently supports:

        * :class:`~pyrite.Ligand`
        * :class:`~pyrite.Receptor`
        * :class:`~pyrite.bounds.Bounds`

        Parameters
        ----------
        *args : Ligand, Receptor, Bounds
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
        # Bugfix: this function is called upon display of the interactive. This is too early,
        # and the viewer is not initialized yet. This adds a new, blank viewer. Using observe doesnt
        # work for some reason.
        if self.__interactive_first is True:
            self.__interactive_first = False
            return
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
        self.view.update()

    def add_v(self, ligand: Ligand, v: NDArray, slider: bool = True, options=None):
        """Adds a ligand with poses, and an optional pose selection slider, to the viewer.

        This method adds a ligand, with poses `v`, to the viewer,
        optionally with draw options `options`. Optionally, a `slider` can be added, which allows
        for the selection of the displayed pose. If `slider` is ``False``, all poses are shown.

        Parameters
        ----------
        ligand : Ligand
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

        self._ligand = ligand
        self._vs = v
        self._v_draw_options = ligand.draw_options.copy()
        self._v_draw_options.update(options)

        if slider:
            mblock = Chem.MolToMolBlock(ligand)
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

            self.__interactive_first = True
            self._interactive = interactive(
                self._set_ligand,
                v_id=IntSlider(
                    min=0,
                    max=len(v) - 1,
                    step=1,
                    continuous_update=True,
                    description="Pose:",
                ),
            )

        else:
            for var in self._vs:
                conf_id = ligand.update(var, new_conf=True)
                mblock = Chem.MolToMolBlock(ligand, confId=conf_id)
                ligand.RemoveConformer(conf_id)
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
        self.view.zoomTo()
        self.view.show()
        if self._interactive is not None:
            display(self._interactive)

        # display(self._repr_html_())

    def _repr_html_(self):
        self.view.zoomTo()
        return self.view.write_html()
