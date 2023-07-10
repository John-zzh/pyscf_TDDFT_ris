#!/bin/bash

for i in $(ls -d */)
do
rm -r $i
done