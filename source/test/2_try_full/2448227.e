--------------------------------------------------------------------------
WARNING: There was an error initializing an OpenFabrics device.

  Local host:   ibc11b02n12
  Local device: mlx5_0
--------------------------------------------------------------------------
[ibc11b02n12:101687] 55 more processes have sent help message help-mpi-btl-openib.txt / error in device init
[ibc11b02n12:101687] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
[ibc11b02n12:101726:0:101726]        rndv.c:1457 Fatal: failed to unpack rendezvous remote key received from <no debug data>: Input/output error

Program received signal SIGSEGV: Segmentation fault - invalid memory reference.

Backtrace for this error:

Could not print backtrace: mmap, errno: 12
#0  0x2b42cdcb95ea
#1  0x2b42cdcb87f5
#2  0x2b42cc2f262f
#3  0x2b42ceb407b4
#4  0x2b42ceb41d71
#5  0x2b42ceb43e11
#6  0x2b42e0be3142
#7  0x2b42e0c2a1bf
#8  0x2b42e0c05885
#9  0x2b42e0bcb12d
#10  0x2b42e0be5beb
#11  0x2b42e0bcb841
#12  0x2b42e0bcba6b
#13  0x2b42e0bcbfe3
#14  0x2b42e0bce3df
#15  0x2b42e0bcac5f
#16  0x2b42e0bcadf8
#17  0x2b42e04dcc5c
#18  0x2b42e04ec59d
#19  0x2b42e073eb55
#20  0x2b42e04b8c79
#21  0x2b42cf3775d2
#22  0x2b42cf37dcc4
#23  0x2b42cc54fe0b
#24  0x2b42cc5ac792
#25  0x2b42e2beddbd
#26  0x2b42cc562b72
#27  0x2b42cda82837
#28  0x2a0d5b4
#29  0xd7afac
#30  0xa61a68
#31  0x8a771b
#32  0x15fe6a0
#33  0x13aa9c5
#34  0x13b718f
#35  0xcf2f44
#36  0xe4b7be
#37  0xe4f19d
#38  0xafe724
#39  0x69ca09
#40  0x68759f
#41  0x6005d8
#42  0x5359a0
#43  0x5384df
#44  0x52a770
#45  0x4be36c
#46  0x2b42ceae0554
#47  0x528d5b
#48  0xffffffffffffffff
--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
mpirun noticed that process rank 29 with PID 101726 on node ibc11b02n12 exited on signal 11 (Segmentation fault).
--------------------------------------------------------------------------
