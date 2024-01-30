import functools
import os.path

import vapoursynth as vs

std = vs.core.std

bs = vs.core.bs
edgefixer = vs.core.edgefixer
fb = vs.core.fb
vivtc = vs.core.vivtc

import awsmfunc as awf
import havsfunc as haf 
import kagefunc as kgf

import vsdehalo
import vsutil

def StrValToBool(val):
	if isinstance(val, bool):
		return val
	if val.lower() in ("true", "1"):
		return True
	elif val.lower() in ("false", "0"):
		return False
	else:
		raise ValueError("value '%s' is not boolean" % val)

def StrValToInt(val):
	if isinstance(val, int):
		return val
	return int(val)

preview = StrValToBool(globals().get('preview', False))
crop_left = StrValToInt(globals().get('crop_left', 0))
#crop_right = StrValToInt(globals().get('crop_right', 0))
crop_right = (720 - 712) - crop_left

# cropping is hard!
# - the standard crop can only crop in amounts divisble by 2
# - while zresize can crop by single pixels, the final dimensions still need to be even
# - can't do the single pixel crop until the end, because filters need the original chroma alignment for best results
# - some filters along the way need to operate on the actual video area, ignoring any remaining black bars
crop_left_true_rem = 0
crop_right_true_rem = 0
if (crop_left + crop_right) % 2 == 1:
	if crop_right:
		crop_right = crop_right - 1
		crop_right_true_rem = 1
	elif crop_left:
		crop_left = crop_left - 1
		crop_left_true_rem = 1
	else:
		assert False, "This should not happen"
crop_left_rem = crop_left % 2
crop_left_true_rem += crop_left_rem
crop_left2 = crop_left - crop_left_rem
crop_right_rem = crop_right % 2
crop_right_true_rem += crop_right_rem
crop_right2 = crop_right - crop_right_rem


scaled_height = 540


def IfCombed(n, f, comb, prog):
	if f.props['_Combed'] > 0:
		return comb
	else:
		return prog


def DeTelecine(clip):
	clip = vivtc.VFM(clip, order=1)
	deint = vs.core.eedi3m.EEDI3(clip, field=1)
	clip = std.FrameEval(
		clip, functools.partial(IfCombed, comb=deint, prog=clip),
		prop_src=clip)
	clip = vivtc.VDecimate(clip)
	return clip


# mitigate some dirty lines around the edges. this is not a perfect solution
# because of how variable these lines are, but it does a good job of reducing
# their impact.
# fix_dist: how far in from detected border of video to fix lines
# border_detect: how far in to scan for the true video borders
def FixEdges(clip, fix_dist=5, border_detect=10):
	# exact size of black borders varies shot-by-shot, so we need to detect where the real edge is
	orig = vs.core.acrop.CropValues(clip, range=border_detect)

	def ProcessFrame(n, f, c, o, crop_left_true_rem, crop_right_true_rem):
		import vapoursynth as vs

		lcrop = max(crop_left_true_rem, f.props['CropLeftValue'])
		rcrop = max(crop_right_true_rem, f.props['CropRightValue'])

		# fill in borders so the black areas don't drag the following blur down
		refbase = vs.core.fb.FillBorders(c, left=lcrop, right=rcrop)

		# do horiz and vert as separate passes to minimize blurring in the wrong direction.
		ref = vs.core.std.BoxBlur(refbase, hradius=1, hpasses=1, vpasses=0)
		c = vs.core.edgefixer.Reference(c, ref, left=fix_dist+lcrop, right=fix_dist+rcrop)
		ref = vs.core.std.BoxBlur(refbase, vradius=1, hpasses=0, vpasses=1)
		c = vs.core.edgefixer.Reference(c, ref, top=fix_dist, bottom=fix_dist)
	
		# now there's a blurry fill in any previously-black bars on the edges, restore the original contents of those bars
		clips = []
		if lcrop > 0:
			clips.append(vs.core.std.BlankClip(width=lcrop, height=c.height, color=1023, format=vs.GRAY10))
		clips.append(vs.core.std.BlankClip(width=c.width - (lcrop + rcrop), height=c.height, format=vs.GRAY10))
		if rcrop > 0:
			clips.append(vs.core.std.BlankClip(width=rcrop, height=c.height, color=1023, format=vs.GRAY10))
		mask = vs.core.std.StackHorizontal(clips)
		return vs.core.std.MaskedMerge(c, o, mask)
	
	clip = std.FrameEval(orig, functools.partial(
		ProcessFrame,
		c=clip,
		o=orig,
		crop_left_true_rem=crop_left_true_rem,
		crop_right_true_rem=crop_right_true_rem
	), prop_src=orig)
	return clip


def PrepOutput(clip):
	true_width = (clip.width - (crop_left_rem + crop_right_rem))
	if true_width % 2:
		clip = std.AddBorders(clip, right=1)

	if preview:
		return awf.zresize(clip, width=clip.width, height=scaled_height, left=crop_left_rem, right=crop_right_rem, format=vs.YUV420P10)
	else:
		return awf.zresize(clip, left=crop_left_rem, right=crop_right_rem, format=vs.YUV420P10)
	

def DoCrop(clip):
	return std.Crop(clip, left=crop_left2, right=crop_right2)


def AlignToCrop(clip):
	# Adds black borders to align the original video with the cropped version when both are centered in the screen.
	return std.AddBorders(
		clip,
		left=max(0, crop_right2-crop_left2),
		right=max(0, crop_left2-crop_right2)
	)
	

src = bs.VideoSource(fn)
if preview:
	#uncropped = PrepOutput(AlignToCrop(DeTelecine(src)))
	#uncropped.set_output(9)
	unfiltered = PrepOutput(DoCrop(DeTelecine(src)))
	unfiltered.set_output(8)
curr = src

curr = vs.core.bifrost.Bifrost(curr, interlaced=True)
curr = vs.core.tcomb.TComb(curr, mode=2)
curr = DeTelecine(curr)

# bifrost and vfm don't allow 10bit so, can't upsample the processing space until now.
curr = vsutil.depth(curr, 10)

# deblock comes before any crop so it can see the original alignment
curr = haf.Deblock_QED(curr)

curr = DoCrop(curr)

curr = FixEdges(curr)

# fill single-pixel black bars
curr = fb.FillBorders(curr,
	left=crop_left_true_rem if (crop_left_true_rem - crop_left_rem) == 1 else 0,
	right=crop_right_true_rem if (crop_right_true_rem - crop_right_rem) == 1 else 0
)

# source has a ton of chroma noise
curr = vs.core.neo_vd.VagueDenoiser(curr, threshold=1, planes=[1,2])
curr = vs.core.neo_fft3d.FFT3D(curr, bt=5, sigma=4, planes=[1,2])

curr = vsdehalo.smooth_dering(curr, mrad=2, minp=4, mthr=0.2, elast=4)

curr = vs.core.neo_fft3d.FFT3D(curr, bt=5, sigma=0.8, planes=[0])
curr = vs.core.neo_vd.VagueDenoiser(curr, threshold=0.8, planes=[0])
# correct the slight blur from the denoise
curr = vs.core.cas.CAS(curr, sharpness=0.3, planes=[0])

curr = kgf.adaptive_grain(curr, static=False)
filtered = PrepOutput(curr)

encfn = fn.replace("remux/", "encode/")
if preview and os.path.exists(encfn):
	enc = bs.VideoSource(encfn)
	enc = awf.zresize(enc, width=enc.width, height=scaled_height)
	enc.set_output(0)
	filtered.set_output(1)
else:
	filtered.set_output(0)
