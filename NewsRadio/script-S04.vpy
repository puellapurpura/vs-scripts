import functools
import os.path

import vapoursynth as vs

import awsmfunc as awf
import havsfunc as haf 
import kagefunc as kgf

import vsdehalo
import vsutil

import funcs

preview = funcs.StrValToBool(globals().get('preview', False))
comparison = funcs.StrValToBool(globals().get('comparison', False))
crop_left = funcs.StrValToInt(globals().get('crop_left', 0))

# narrower: 2,3 ; 6,7,8,9,10,12,14,15,16
ep = int(fn.split(" - ")[1].split("E")[1])
if ep in (1, 4, 5, 11, 13, 17, 18, 19, 20, 22):
	crop_right = (720 - 712) - crop_left
elif ep in (10, 21):
	crop_right = (720 - 710) - crop_left
elif ep in (2, 3, 6, 7, 8, 9, 12, 14, 15):
	crop_right = (720 - 708) - crop_left
elif ep in (16,):
	crop_right = (720 - 706) - crop_left
else:
	raise Exception("Episode crop parameters not set!")

crop = funcs.Border(left=crop_left, right=crop_right)


scaled_height = 540

output_format = vs.YUV420P10

VideoSource = functools.partial(
	vs.core.bs.VideoSource,
	cachepath=os.path.expanduser("~/.bsindex.json")
)


def PrepOutput(clip, crop):
	kw = {}
	if preview or comparison:
		kw['scale_height'] = scaled_height
	
	return funcs.ZresizeCrop(clip, crop, zresize_kwargs={'format': output_format}, **kw)


src = VideoSource(fn)
if preview:
	uncropped = funcs.AlignToZresizeCrop(
		funcs.RestoreProgressive(src),
		crop,
		scale_height=scaled_height,
		zresize_kwargs={'format': output_format})
	uncropped.set_output(9)
if preview or comparison:
	unfiltered, _ = PrepOutput(funcs.RestoreProgressive(src), crop)
	unfiltered.set_output(8)
curr = src

curr = vs.core.bifrost.Bifrost(curr, interlaced=True)
curr = vs.core.tcomb.TComb(curr, mode=2, fthreshl=2, othreshl=2, fthreshc=6, othreshc=6)
curr = funcs.RestoreProgressive(curr)

# bifrost and vfm don't allow 10bit so, can't upsample until now.
curr = vsutil.depth(curr, 10)

# deblock comes before any crop so it can see the original alignment
curr = haf.Deblock_QED(curr)

curr, remcrop = funcs.SafeCrop(curr, crop)


curr = funcs.FixEdges(curr, fix_vert=0, static_border=remcrop)

curr = vs.core.neo_vd.VagueDenoiser(curr, threshold=0.4, planes=[1,2])
curr = vs.core.neo_fft3d.FFT3D(curr, bt=5, sigma=1.9, planes=[1,2])

curr = vsdehalo.smooth_dering(curr, mrad=2, minp=4, mthr=0.2, elast=4)

curr = vs.core.neo_fft3d.FFT3D(curr, bt=5, sigma=0.4, planes=[0])
curr = vs.core.neo_vd.VagueDenoiser(curr, threshold=0.1, planes=[0])
# correct the slight blur from the denoise
curr = vs.core.cas.CAS(curr, sharpness=0.07, planes=[0])

curr = kgf.adaptive_grain(curr, static=False)
filtered, _ = PrepOutput(curr, remcrop)

encfn = fn.replace("remux/", "encode/")
if comparison and os.path.exists(encfn):
	enc = VideoSource(encfn)
	enc = awf.zresize(enc, width=enc.width, height=scaled_height)
	enc.set_output(0)
	filtered.set_output(1)
else:
	filtered.set_output(0)
