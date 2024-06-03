--------------------------------------------------------------------------
WARNING: There was an error initializing an OpenFabrics device.

  Local host:   ibc11b02n08
  Local device: mlx5_0
--------------------------------------------------------------------------
[ibc11b02n08:58997] 55 more processes have sent help message help-mpi-btl-openib.txt / error in device init
[ibc11b02n08:58997] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
[ibc11b02n08:59025:0:59025]        rndv.c:1457 Fatal: failed to unpack rendezvous remote key received from <no debug data>: Input/output error

Program received signal SIGSEGV: Segmentation fault - invalid memory reference.

Backtrace for this error:

Could not print backtrace: mmap, errno: 12
#0  0x2b238ff105ea
#1  0x2b238ff0f7f5
#2  0x2b238e54962f
#3  0x2b2390d97272
#4  0x2b2390d98d71
#5  0x2b2390d9ae11
#6  0x2b23a2d39142
#7  0x2b23a2d801bf
#8  0x2b23a2d5b885
#9  0x2b23a2d2112d
#10  0x2b23a2d3bbeb
#11  0x2b23a2d21841
#12  0x2b23a2d21a6b
#13  0x2b23a2d21fe3
#14  0x2b23a2d243df
#15  0x2b23a2d20c5f
#16  0x2b23a2d20df8
#17  0x2b23a2632c5c
#18  0x2b23a264259d
#19  0x2b23a2894b55
#20  0x2b23a260ec79
#21  0x2b23915ce5d2
#22  0x2b23915d4cc4
#23  0x2b238e7a69ab
#24  0x2b238e7e68fa
#25  0x2b238fce3969
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
#47  0x2b2390d37554
#48  0x528d5b
#49  0xffffffffffffffff
--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
mpirun noticed that process rank 28 with PID 59025 on node ibc11b02n08 exited on signal 11 (Segmentation fault).
--------------------------------------------------------------------------
