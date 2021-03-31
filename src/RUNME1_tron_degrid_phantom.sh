#!/bin/sh
make &&
{
  echo ------- degridding Shepp-Logan phantom using TRON
  ./tron -i 3 ../data/shepplogan.ra output/sl_data_tron.ra
} 
