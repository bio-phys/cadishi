# Bash command completion for cadishi.
#
# `source` this file into a bash session (e.g. automatically using .bashrc)
# and use the TAB key to get command suggestions.
#
# This file is part of the Cadishi package.  See README.rst,
# LICENSE.txt, and the documentation for details.

_cadishi()
{
    local cur prev opts histo_opts example_opts merge_opts unpack_opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    opts="histo example merge unpack --help --version"
    histo_opts="--help --input"
    example_opts="--help --output"
    merge_opts="--help --force --output"
    unpack_opts="--help --force --output"

    if [[ ${prev} == --input ]] ; then
        COMPREPLY=( $(compgen -f -X '!*.yaml' -- ${cur}) )
        return 0
    fi
    if [[ ${prev} == --output ]] ; then
        COMPREPLY=( $(compgen -f -X '!*.yaml' -- ${cur}) )
        return 0
    fi
    if [[ ${prev} == histo ]] ; then
        COMPREPLY=( $(compgen -W "${histo_opts}" -- ${cur}) )
        return 0
    fi
    if [[ ${prev} == example ]] ; then
        COMPREPLY=( $(compgen -W "${example_opts}" -- ${cur}) )
        return 0
    fi
    if [[ ${prev} == merge ]] ; then
        COMPREPLY=( $(compgen -W "${merge_opts}" -- ${cur}) )
        return 0
    fi
    if [[ ${prev} == unpack ]] ; then
        COMPREPLY=( $(compgen -W "${unpack_opts}" -- ${cur}) )
        return 0
    fi
    if [[ ${cur} == * ]] ; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
}
complete -F _cadishi cadishi
