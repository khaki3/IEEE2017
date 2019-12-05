#!/bin/bash

{
    for i in */*/_run.sh; do
        echo ==== ./$i; (cd `dirname $i`; ./`basename $i`);
    done
} | tee stencilgen.log
