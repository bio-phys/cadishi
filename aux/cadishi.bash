# -*- tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

# Bash command completion for cadishi.
#
# `source` this file into a bash session (e.g. automatically using .bashrc)
# and use the TAB key to get command suggestions.

_cadishi()
{
    local cur prev opts histo_opts example_opts merge_opts unpack_opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    opts="histo example merge unpack --help --version"
    histo_opts="--help"
    example_opts="--help --expert"
    merge_opts="--help --force --output --compression"
    unpack_opts="--help --force --output"

    if [[ ${prev} == --compression ]] ; then
        COMPREPLY=( $(compgen -W "${compression_opts}" -- ${cur}) )
        return 0
    fi
    if [[ ${prev} == histo ]] ; then
        COMPREPLY=( $(compgen -W "${histo_opts}" -- ${cur}) $(compgen -f -X '!*.yaml' -- ${cur}) )
        return 0
    fi
    if [[ ${prev} == example ]] ; then
        COMPREPLY=( $(compgen -W "${example_opts}" -- ${cur}) )
        return 0
    fi
    if [[ ${prev} == merge ]] ; then
        COMPREPLY=( $(compgen -W "${merge_opts}" -- ${cur}) $(compgen -f -X '!*.h5' -- ${cur}) )
        return 0
    fi
    if [[ ${prev} == unpack ]] ; then
        COMPREPLY=( $(compgen -W "${unpack_opts}" -- ${cur}) $(compgen -f -X '!*.h5' -- ${cur}) )
        return 0
    fi
    if [[ ${cur} == * ]] ; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
}
complete -F _cadishi cadishi
