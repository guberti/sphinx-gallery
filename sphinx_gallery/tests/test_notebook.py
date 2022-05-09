# -*- coding: utf-8 -*-
# Author: Óscar Nájera
# License: 3-clause BSD
r"""
Testing the Jupyter notebook parser
"""

from __future__ import division, absolute_import, print_function
import json
import tempfile
import os
import pytest
import textwrap

import sphinx_gallery.gen_rst as sg
from sphinx_gallery.notebook import (rst2md, jupyter_notebook, save_notebook,
                                     promote_jupyter_cell_magic,
                                     python_to_jupyter_cli)
from sphinx_gallery.tests.test_gen_rst import gallery_conf  # noqa

try:
    FileNotFoundError
except NameError:
    # Python2
    FileNotFoundError = IOError


def test_latex_conversion():
    """Latex parsing from rst into Jupyter Markdown"""
    double_inline_rst = r":math:`T<0` and :math:`U>0`"
    double_inline_jmd = r"$T<0$ and $U>0$"
    assert double_inline_jmd == rst2md(double_inline_rst)

    align_eq = r"""
.. math::
   \mathcal{H} &= 0 \\
   \mathcal{G} &= D"""

    align_eq_jmd = r"""
\begin{align}\mathcal{H} &= 0 \\
   \mathcal{G} &= D\end{align}"""
    assert align_eq_jmd == rst2md(align_eq)


def test_code_conversion():
    """Use the ``` code format so Jupyter syntax highlighting works"""
    rst = textwrap.dedent("""
        Regular text
            .. code-block::

               # Bash code

          More regular text
        .. code-block:: cpp

          //cpp code

          //more cpp code
        non-indented code blocks are not valid
        .. code-block:: cpp

        // not a real code block
    """)
    assert rst2md(rst) == textwrap.dedent("""
        Regular text
        ```
        # Bash code
        ```
          More regular text
        ```cpp
        //cpp code

        //more cpp code
        ```
        non-indented code blocks are not valid
        .. code-block:: cpp

        // not a real code block
    """)


def test_convert():
    """Test ReST conversion"""
    rst = """hello

.. contents::
    :local:

This is :math:`some` math :math:`stuff`.

.. note::
    Interpolation is a linear operation that can be performed also on
    Raw and Epochs objects.

.. warning::
    Go away

For more details on interpolation see the page :ref:`channel_interpolation`.
.. _foo: bar

`See more  <https://en.wikipedia.org/wiki/Interpolation>`_.
"""

    markdown = """hello

This is $some$ math $stuff$.

<div class="alert alert-info"><h4>Note</h4><p>Interpolation is a linear operation that can be performed also on
    Raw and Epochs objects.</p></div>

<div class="alert alert-danger"><h4>Warning</h4><p>Go away</p></div>

For more details on interpolation see the page `channel_interpolation`.

[See more](https://en.wikipedia.org/wiki/Interpolation).
"""  # noqa
    assert rst2md(rst) == markdown


def test_cell_magic_promotion():
    markdown = textwrap.dedent("""\
    # Should be rendered as text
    ``` bash
    # This should be rendered as normal
    ```
    ``` bash
    %%bash
    # bash magic
    ```
    ```cpp
    %%writefile out.cpp
    // This c++ cell magic will write a file
    // There should NOT be a text block above this
    ```
    Interspersed text block
    ```javascript
    %%javascript
    // Should also be a code block
    // There should NOT be a trailing text block after this
    ```
    """)
    work_notebook = {"cells": []}
    promote_jupyter_cell_magic(work_notebook, markdown)
    cells = work_notebook["cells"]

    assert len(cells) == 5
    assert cells[0]["cell_type"] == "markdown"
    assert "``` bash" in cells[0]["source"][0]
    assert cells[1]["cell_type"] == "code"
    assert cells[1]["source"][0] == "%%bash\n# bash magic"
    assert cells[2]["cell_type"] == "code"
    assert cells[3]["cell_type"] == "markdown"
    assert cells[3]["source"][0] == "Interspersed text block"
    assert cells[4]["cell_type"] == "code"


def test_jupyter_notebook(gallery_conf):
    """Test that written ipython notebook file corresponds to python object"""
    file_conf, blocks = sg.split_code_and_text_blocks('tutorials/plot_parse.py')
    example_nb = jupyter_notebook(blocks, gallery_conf)

    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        save_notebook(example_nb, f.name)
    try:
        with open(f.name, "r") as fname:
            assert json.load(fname) == example_nb
    finally:
        os.remove(f.name)
    assert example_nb.get('cells')[0]['source'][0] == "%matplotlib inline"

    # Test custom first cell text
    test_text = '# testing\n%matplotlib notebook'
    gallery_conf['first_notebook_cell'] = test_text
    example_nb = jupyter_notebook(blocks, gallery_conf)
    assert example_nb.get('cells')[0]['source'][0] == test_text

    # Test empty first cell text
    test_text = None
    gallery_conf['first_notebook_cell'] = test_text
    example_nb = jupyter_notebook(blocks, gallery_conf)
    assert example_nb.get('cells')[0]['source'][0].startswith('\nThe Header Docstring')

###############################################################################
# Notebook shell utility


def test_with_empty_args():
    """ User passes no args, should fail with SystemExit """
    with pytest.raises(SystemExit):
        python_to_jupyter_cli([])


def test_missing_file():
    """ User passes non existing file, should fail with FileNotFoundError """
    with pytest.raises(FileNotFoundError) as excinfo:
        python_to_jupyter_cli(['nofile.py'])
    excinfo.match(r'No such file or directory.+nofile\.py')


def test_file_is_generated():
    """User passes good python file. Check notebook file is created"""

    python_to_jupyter_cli(['examples/plot_quantum.py'])
    assert os.path.isfile('examples/plot_quantum.ipynb')
    os.remove('examples/plot_quantum.ipynb')
