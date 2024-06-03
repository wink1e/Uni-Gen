--------------------------------------------------------------------------
WARNING: There was an error initializing an OpenFabrics device.

  Local host:   ibc11b02n05
  Local device: mlx5_0
--------------------------------------------------------------------------
[ibc11b02n05:43068] 55 more processes have sent help message help-mpi-btl-openib.txt / error in device init
[ibc11b02n05:43068] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
[ibc11b02n05:43106:0:43106]        rndv.c:1457 Fatal: failed to unpack rendezvous remote key received from <no debug data>: Input/output error

Program received signal SIGSEGV: Segmentation fault - invalid memory reference.

Backtrace for this error:

Could not print backtrace: mmap, errno: 12
#0  0x2ba2b8b5e5ea
#1  0x2ba2b8b5d7f5
#2  0x2ba2b719762f
#3  0x2ba2b99e5272
#4  0x2ba2b99e878b
#5  0x2ba2cb9490bd
#6  0x2ba2cb9d1eb7
#7  0x2ba2cb9d86ed
#8  0x2ba2cb98ad6c
#9  0x2ba2cb98cdaa
#10  0x2ba2cb990614
#11  0x2ba2cb96b885
#12  0x2ba2cb93112d
#13  0x2ba2cb94bbeb
#14  0x2ba2cb931841
#15  0x2ba2cb931a6b
#16  0x2ba2cb931fe3
#17  0x2ba2cb9343df
#18  0x2ba2cb930c5f
#19  0x2ba2cb930df8
#20  0x2ba2cb242c5c
#21  0x2ba2cb25259d
#22  0x2ba2cb4a4b55
#23  0x2ba2cb21ec79
#24  0x2ba2ba21c5d2
#25  0x2ba2ba222cc4
#26  0x2ba2b73f49ab
#27  0x2ba2b74348fa
#28  0x2ba2b8931969
#29  0x2a3a1fe
#30  0x26f21a0
#31  0x26f3d39
#32  0x2687b76
#33  0xa61acc
#34  0x8a771b
#35  0x15fe6a0
#36  0x13aa9c5
#37  0x13b718f
#38  0xcf2f44
#39  0xe4b7be
#40  0xe4f19d
#41  0xafe724
#42  0x69ca09
#43  0x68759f
#44  0x6005d8
#45  0x5359a0
#46  0x5384df
#47  0x52a770
#48  0x4be36c
#49  0x2ba2b9985554
#50  0x528d5b
#51  0xffffffffffffffff
--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
mpirun noticed that process rank 29 with PID 43106 on node ibc11b02n05 exited on signal 11 (Segmentation fault).
--------------------------------------------------------------------------
