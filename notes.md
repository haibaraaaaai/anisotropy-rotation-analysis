2 things?

guard: need to make sure if ref cycle is done at say 60-61s, it's somehow still accounted for for 0s data? Maybe we should just start from 0-1s data anyway? How is it handled now? Need to make sure every time guard is triggered it's printed. (I think it is)

phase on ref cycle are uniformly spaced in angles in rotation frame.
So in theory for the phase should be more evenly distributed (maybe not between phase to phase, and maybe there will be some preferred angles, but overall we should expect some sort of uniformity for how long it takes to travel between phases)
Maybe this is something to plot? maybe we can add additional smoothing / correction to the speed base on this? i.e. smooth the speed so that it covers all phases at about the same rate, or just take per rev speed like i used to do.

Or maybe it can be used as a way to verify or even go into shape fitting as some sort of quantity to fit over?

For now the per rev speed is probably just the thing to do..


4. regarding wobbles / noise / drifts... And this is the main thing I want to discuss, but perhaps we can do that later after we tried your method and make sure the projection is working as intended. The purpose of speed measurement is to use that to study stator dynamics, when does stator unit number change, is multiple stator speed just single stator speed times N? do we have a weakly bound state for stator unit where it might only contribute to torque and speed for a second or even less, and if so can we capture it. So a lot of the work I think is about identifying what in the speed is real (i.e. stator dynamics), what is from the electronic noise of the APDs, what is from the wobbles, not sure exactly where the wobble comes from but it is not what we care? And maybe if there's anything between PMF and torque generation that also create speed changes but it's not about stator units... (which is getting to the more physics side) And the key there I think is to understand the timescales, or try to estimate the timescales base on data, and smooth over the irrelavent parts. Because as you will surely discover when we do some testing of your projection, is that the speed varies A LOT and very quickly. So we need to find ways to get closer to the true speed if we can.

126891 127550
need to store indices for manual in teh output somewhere (just print it) so it's fine when kernel restarts / some funny things happened

ok steps are actually already done by my mentor, if you mean the 26 steps per rev resolution, it's done with passive rotation (i.e. no stator units) so we have much clearer steps. I can show you content of that paper if it becomes relevant. What I'm curious is 


region 230-260s : 300 speed samples @ dt=100 ms
penalty = 1140.8  ->  25 states detected

 #   t_start     t_end   level/Hz   dwell/s
 0     230.1     230.2      161.1      0.20
 1     230.3     230.9      192.3      0.70
 2     231.0     232.0      139.4      1.10
 3     232.1     232.1       58.8      0.10
 4     232.2     233.3      105.3      1.20
 5     233.4     233.7       61.5      0.40
 6     233.8     234.3       31.2      0.60
 7     234.4     235.7      105.8      1.40
 8     235.8     236.3       86.6      0.60
 9     236.4     237.8      114.9      1.50
10     237.9     238.9       86.3      1.10
11     239.0     239.8       61.6      0.90
12     239.9     240.0       18.1      0.20
13     240.1     240.1       68.9      0.10
14     240.2     240.8      114.5      0.70
15     240.9     241.0       78.1      0.20
16     241.1     241.7      125.5      0.70
17     241.8     243.6      151.8      1.90
18     243.7     243.7       90.5      0.10
19     243.8     246.9      139.4      3.20
20     247.0     247.0       75.6      0.10
21     247.1     247.4      122.3      0.40
22     247.5     251.3      145.9      3.90
23     251.4     259.8      225.3      8.50
24     259.9     260.0      153.0      0.20

step sizes between consecutive states (Hz):
  +31.2, -52.9, -80.7, +46.5, -43.8, -30.3, +74.6, -19.2, +28.3, -28.6, -24.7, -43.5, +50.7, +45.7, -36.5, +47.4, +26.4, -61.4, +48.9, -63.8, +46.7, +23.6, +79.4, -72.3

|step| stats: median=46.1 Hz, max=80.7 Hz  (single-stator quantum ~50 Hz)