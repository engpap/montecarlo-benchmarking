# MonteCarlo Benchmarking
This is a project of the [High Performance Processors and Systems](https://www4.ceda.polimi.it/manifesti/manifesti/controller/ManifestoPublic.do?EVN_DETTAGLIO_RIGA_MANIFESTO=evento&aa=2022&k_cf=225&k_corso_la=481&k_indir=T2A&codDescr=089185&lang=EN&semestre=2&idGruppo=4474&idRiga=281811) course of Politecnico Di Milano.

This repository contains the evaluation and implementation of MonteCarlo workload on multi-GPU systems via Unified Memory.
The project is based on the [Tartan benchmarking suite](https://github.com/uuudown/Tartan/blob/master/IISWC-18.pdf)


## Team
* [Andrea Paparella](https://github.com/engpap)
* [Andrea Piras](https://github.com/andreapiras00)

## Useful Commands
```conda deactivate```<br />
```conda activate``` <br />
<br />
Nsight Systems:<br />

```nsys profile -o <report_file_name> --stats=true ./MonteCarlo --method=<method> --scaling=<scaling>```<br />
```--stats=true``` to generate a summary of CPU and GPU activities
or<br />
```nsys profile ./MonteCarlo```<br />
