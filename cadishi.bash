# Bash command completion for cadishi.
#
# `source` this file into a bash session (e.g. automatically using .bashrc)
# and use the TAB key to get command suggestions.
#
# This file is part of the Cadishi package.  See README.rst,
# LICENSE.txt, and the documentation for details.

_cadishi()
{
    local cur prev opts histo_opts histo_example_opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    opts="histo histo-example -h --help"
    histo_opts="-h --help --input"
    histo_example_opts="-h --help --output"

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
    if [[ ${prev} == histo-example ]] ; then
        COMPREPLY=( $(compgen -W "${histo_example_opts}" -- ${cur}) )
        return 0
    fi
    if [[ ${cur} == * ]] ; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
}
complete -F _cadishi cadishi
