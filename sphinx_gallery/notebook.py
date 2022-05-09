# -*- coding: utf-8 -*-
r"""
Parser for Jupyter notebooks
============================

Class that holds the Jupyter notebook information

"""
# Author: Óscar Nájera
# License: 3-clause BSD

from __future__ import division, absolute_import, print_function
from functools import partial
import argparse
import copy
import json
import re
import sys
import textwrap

from .py_source_parser import split_code_and_text_blocks
from .utils import replace_py_ipynb


def jupyter_notebook_skeleton():
    """Returns a dictionary with the elements of a Jupyter notebook"""
    py_version = sys.version_info
    notebook_skeleton = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python " + str(py_version[0]),
                "language": "python",
                "name": "python" + str(py_version[0])
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": py_version[0]
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython" + str(py_version[0]),
                "version": '{0}.{1}.{2}'.format(*sys.version_info[:3])
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    return notebook_skeleton


def directive_fun(match, directive):
    """Helper to fill in directives"""
    directive_to_alert = dict(note="info", warning="danger")
    return ('<div class="alert alert-{0}"><h4>{1}</h4><p>{2}</p></div>'
            .format(directive_to_alert[directive], directive.capitalize(),
                    match.group(1).strip()))


def convert_code_to_md(text):
    """Rewrites code blocks using the code-block notation to use Markdown's
    preferred backtick notation, which preserves syntax highlighting.

    Parameters
    ----------
    text: str
        A mostly converted string of markdown text. May contain zero, one,
        or multiple code blocks in code-block format.
    """

    code_regex = r'[ \t]*\.\. code-block::[ \t]*(\S*)\n[ \t]*\n([ \t]+)'
    while True:
        code_block = re.search(code_regex, text)
        if not code_block:
            break
        indent = code_block.group(2)
        start_index = code_block.span()[1] - len(indent)

        # Find first non-empty, non-indented line
        end = re.compile(fr'^(?!{re.escape(indent)})[ \t]*\S+', re.MULTILINE)
        code_end_match = end.search(text, start_index)
        end_index = code_end_match.start() if code_end_match else len(text)

        contents = textwrap.dedent(text[start_index:end_index]).rstrip()
        new_code = (f'```{code_block.group(1)}\n{contents}\n```\n')
        text = text[:code_block.span()[0]] + new_code + text[end_index:]
    return text


def rst2md(text):
    """Converts the RST text from the examples docstrigs and comments
    into markdown text for the Jupyter notebooks"""

    top_heading = re.compile(r'^=+$\s^([\w\s-]+)^=+$', flags=re.M)
    text = re.sub(top_heading, r'# \1', text)

    math_eq = re.compile(r'^\.\. math::((?:.+)?(?:\n+^  .+)*)', flags=re.M)
    text = re.sub(math_eq,
                  lambda match: r'\begin{{align}}{0}\end{{align}}'.format(
                      match.group(1).strip()),
                  text)
    inline_math = re.compile(r':math:`(.+?)`', re.DOTALL)
    text = re.sub(inline_math, r'$\1$', text)

    directives = ('warning', 'note')
    for directive in directives:
        directive_re = re.compile(r'^\.\. %s::((?:.+)?(?:\n+^  .+)*)'
                                  % directive, flags=re.M)
        text = re.sub(directive_re,
                      partial(directive_fun, directive=directive), text)

    footnote_links = re.compile(r'^ *\.\. _.*:.*$\n', flags=re.M)
    text = re.sub(footnote_links, '', text)

    embedded_uris = re.compile(r'`([^`]*?)\s*<([^`]*)>`_')
    text = re.sub(embedded_uris, r'[\1](\2)', text)

    refs = re.compile(r':ref:`')
    text = re.sub(refs, '`', text)

    contents = re.compile(r'^\s*\.\. contents::.*$(\n +:\S+: *$)*\n',
                          flags=re.M)
    text = re.sub(contents, '', text)

    images = re.compile(
        r'^\.\. image::(.*$)(?:\n *:alt:(.*$)\n)?(?: +:\S+:.*$\n)*',
        flags=re.M)
    text = re.sub(
        images, lambda match: '![{1}]({0})\n'.format(
            match.group(1).strip(), (match.group(2) or '').strip()), text)

    text = convert_code_to_md(text)

    return text


def jupyter_notebook(script_blocks, gallery_conf):
    """Generate a Jupyter notebook file cell-by-cell

    Parameters
    ----------
    script_blocks : list
        Script execution cells.
    gallery_conf : dict
        The sphinx-gallery configuration dictionary.
    """
    first_cell = gallery_conf.get("first_notebook_cell", "%matplotlib inline")
    work_notebook = jupyter_notebook_skeleton()
    if first_cell is not None:
        add_code_cell(work_notebook, first_cell)
    fill_notebook(work_notebook, script_blocks)

    return work_notebook


def add_code_cell(work_notebook, code):
    """Add a code cell to the notebook

    Parameters
    ----------
    code : str
        Cell content
    """

    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"collapsed": False},
        "outputs": [],
        "source": [code.strip()]
    }
    work_notebook["cells"].append(code_cell)


def add_markdown_cell(work_notebook, text):
    """Add a markdown cell to the notebook

    Parameters
    ----------
    code : str
        Cell content
    """
    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [rst2md(text)]
    }
    work_notebook["cells"].append(markdown_cell)


def promote_jupyter_cell_magic(work_notebook, markdown):
    """Parses a block of markdown text looking for code blocks starting with a
    Jupyter cell magic (e.g. %%bash). Whenever one is found, the text before it
    and the code (as a runnable code block) are added to work_notebook. Any
    remaining text is returned.

    Parameters
    ----------
    markdown : str
        Markdown cell content.
    """

    # Regex detects all code blocks that use %% Jupyter cell magic
    cell_magic_regex = r'\n?```\s*[a-z]*\n(%%(?:[\s\S]*?))\n?```\n?'

    text_cell_start = 0
    for magic_cell in re.finditer(cell_magic_regex, markdown):
        # Extract the preceeding text block, and add it if non-empty
        text_block = markdown[text_cell_start:magic_cell.span()[0]]
        if text_block and not text_block.isspace():
            add_markdown_cell(work_notebook, text_block)
        text_cell_start = magic_cell.span()[1]

        code_block = magic_cell.group(1)
        add_code_cell(work_notebook, code_block)

    # Return remaining text (which equals markdown if no magic cells exist)
    return markdown[text_cell_start:]


def fill_notebook(work_notebook, script_blocks):
    """Writes the Jupyter notebook cells

    Parameters
    ----------
    script_blocks : list
        Each list element should be a tuple of (label, content, lineno).
    """

    for blabel, bcontent, lineno in script_blocks:
        if blabel == 'code':
            add_code_cell(work_notebook, bcontent)
        else:
            remaining = promote_jupyter_cell_magic(work_notebook, bcontent)
            if remaining and not remaining.isspace():
                add_markdown_cell(work_notebook, remaining)


def save_notebook(work_notebook, write_file):
    """Saves the Jupyter work_notebook to write_file"""
    with open(write_file, 'w') as out_nb:
        json.dump(work_notebook, out_nb, indent=2)


###############################################################################
# Notebook shell utility

def python_to_jupyter_cli(args=None, namespace=None):
    """Exposes the jupyter notebook renderer to the command line

    Takes the same arguments as ArgumentParser.parse_args
    """
    from . import gen_gallery  # To avoid circular import
    parser = argparse.ArgumentParser(
        description='Sphinx-Gallery Notebook converter')
    parser.add_argument('python_src_file', nargs='+',
                        help='Input Python file script to convert. '
                        'Supports multiple files and shell wildcards'
                        ' (e.g. *.py)')
    args = parser.parse_args(args, namespace)

    for src_file in args.python_src_file:
        file_conf, blocks = split_code_and_text_blocks(src_file)
        print('Converting {0}'.format(src_file))
        gallery_conf = copy.deepcopy(gen_gallery.DEFAULT_GALLERY_CONF)
        example_nb = jupyter_notebook(blocks, gallery_conf)
        save_notebook(example_nb, replace_py_ipynb(src_file))
