--------------------------------------------------------------------------
WARNING: There was an error initializing an OpenFabrics device.

  Local host:   ibc11b07n27
  Local device: mlx5_0
--------------------------------------------------------------------------
[ibc11b07n27:81518] 55 more processes have sent help message help-mpi-btl-openib.txt / error in device init
[ibc11b07n27:81518] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
[ibc11b07n27:81560:0:81560]        rndv.c:1457 Fatal: failed to unpack rendezvous remote key received from <no debug data>: Input/output error

Program received signal SIGSEGV: Segmentation fault - invalid memory reference.

Backtrace for this error:

Could not print backtrace: mmap, errno: 12
#0  0x2b38c56215ea
#1  0x2b38c56207f5
#2  0x2b38c3c5a62f
#3  0x2b38c64a8080
#4  0x2b38c64a9d71
#5  0x2b38c64abe11
#6  0x2b38d8555142
#7  0x2b38d859c1bf
#8  0x2b38d8577885
#9  0x2b38d853d12d
#10  0x2b38d8557beb
#11  0x2b38d853d841
#12  0x2b38d853da6b
#13  0x2b38d853dfe3
#14  0x2b38d85403df
#15  0x2b38d853cc5f
#16  0x2b38d853cdf8
#17  0x2b38d8066c5c
#18  0x2b38d807659d
#19  0x2b38d82c8b55
#20  0x2b38d8042c79
#21  0x2b38c6cdf5d2
#22  0x2b38c6ce5cc4
#23  0x2b38c3eb79ab
#24  0x2b38c3ef78fa
#25  0x2b38c53f4969
#26  0x2a3a1fe
#27  0x26efdad
#28  0x26f3d39
#29  0x2686770
#30  0x15d4ef5
#31  0x129a9d3
#32  0xcdc71e
#33  0xce1427
#34  0x13aaa7a
#35  0x13b718f
#36  0xcf2f44
#37  0xe4b7be
#38  0xe4f19d
#39  0xafe724
#40  0x69ca09
#41  0x68759f
#42  0x6005d8
#43  0x5359a0
#44  0x5384df
#45  0x52a770
#46  0x4be36c
#47  0x2b38c6448554
#48  0x528d5b
#49  0xffffffffffffffff
--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
mpirun noticed that process rank 29 with PID 81560 on node ibc11b07n27 exited on signal 11 (Segmentation fault).
--------------------------------------------------------------------------
