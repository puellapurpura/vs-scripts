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
import vsfieldkit
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


def RestoreProgressive(clip):
	clip = vivtc.VFM(clip, order=1)
	deint = vs.core.eedi3m.EEDI3(clip, field=1)
	clip = std.FrameEval(
		clip, functools.partial(IfCombed, comb=deint, prog=clip),
		prop_src=clip)
	clip = vsfieldkit.resample_as_progressive(clip)
	return clip


# returns the grayscale equivalent of clip's format, suitable for use as a mask
def MatchingGray(clip):
	import vapoursynth as vs
	return getattr(vs, "GRAY%s" % clip.format.bits_per_sample)


# merges the borders of border_clip onto clip
def MergeBorders(clip, border_clip, left=0, right=0, top=0, bottom=0):
	import functools
	import vapoursynth as vs
	std = vs.core.std
	Black = functools.partial(
		std.BlankClip, format=MatchingGray(clip))
	White = functools.partial(
		Black, color=(2**clip.format.bits_per_sample - 1))
	if left > 0 or right > 0:
		clips = []
		if left > 0:
			clips.append(White(width=left, height=clip.height))
		clips.append(Black(
			width=clip.width - (left + right),
			height=clip.height))
		if right > 0:
			clips.append(White(width=right, height=clip.height))
		mask = std.StackHorizontal(clips)
		clip = std.MaskedMerge(clip, border_clip, mask)
	if top > 0 or bottom > 0:
		clips = []
		if top > 0:
			clips.append(White(height=top, width=clip.width))
		clips.append(Black(
			height=clip.height - (top + bottom),
			width=clip.width))
		if bottom > 0:
			clips.append(White(height=bottom, width=clip.width))
		mask = std.StackVertical(clips)
		clip = std.MaskedMerge(clip, border_clip, mask)
	return clip


# mitigate some dirty lines around the edges. this is not a perfect solution
# because of how variable these lines are, but it does a good job of reducing
# their impact.
# fix_dist: how far in from detected border of video to fix lines
# fix_horiz: like fix_dist but for left/right only
# fix_vert: like fix_dist but for top/bottom only
# border_detect: how far in to scan for the variable video borders
# static_crop_*: known static black borders to compensate for
def FixEdges(clip, fix_dist=None, border_detect=10,
		fix_horiz=None, fix_vert=None,
		static_crop_left=0, static_crop_right=0,
		static_crop_top=0, static_crop_bottom=0):
	if fix_dist is None:
		fix_dist = 6
	if not any((fix_dist, fix_horiz, fix_vert)):
		return clip
	if fix_horiz is None:
		fix_horiz = fix_dist
	if fix_vert is None:
		fix_vert = fix_dist

	# fill in static borders so any noise in the bars doesn't affect detections
	clip = MergeBorders(
		clip,
		std.BlankClip(clip),
		left=static_crop_left,
		right=static_crop_right,
		top=static_crop_top,
		bottom=static_crop_bottom
	)
	# FIXME: conditionalize use of _Even on presence of chroma subsampling???
	# exact size of black borders varies shot-by-shot, so we need to detect
	# where the real edge is
	def _Even(num):
		return num - (num % 2)
	detect = vs.core.acrop.CropValues(
		clip,
		left=_Even(border_detect+static_crop_left),
		right=_Even(border_detect+static_crop_right),
		top=_Even(border_detect+static_crop_top),
		bottom=_Even(border_detect+static_crop_bottom),
		color=[0,118,118],
		color_second=[36,138,138]
	)

	def ProcessFrame(n, f, clip, lcrop, rcrop, tcrop, bcrop, mb, fix_horiz, fix_vert):
		import vapoursynth as vs

		orig = clip

		lcrop = max(lcrop, f.props['CropLeftValue'])
		rcrop = max(rcrop, f.props['CropRightValue'])
		tcrop = max(tcrop, f.props['CropTopValue'])
		bcrop = max(bcrop, f.props['CropBottomValue'])

		# fill in borders so the black areas don't dim the blur
		refbase = vs.core.fb.FillBorders(
			clip, mode="mirror",
			left=lcrop, right=rcrop, top=tcrop, bottom=bcrop)

		# do horiz and vert as separate passes to minimize blurring
		# in the wrong direction.
		if fix_horiz:
			ref = vs.core.std.BoxBlur(
				refbase, hradius=1, hpasses=1, vpasses=0)
			clip = vs.core.edgefixer.Reference(
				clip, ref,
				left=fix_horiz+lcrop, right=fix_horiz+rcrop)
		if fix_vert:
			ref = vs.core.std.BoxBlur(
				refbase, vradius=1, hpasses=0, vpasses=1)
			clip = vs.core.edgefixer.Reference(
				clip, ref,
				top=fix_vert+tcrop, bottom=fix_vert+bcrop)

		# restore borders
		clip = MergeBorders(clip, orig, lcrop, rcrop, tcrop, bcrop)
		return clip

	clip = std.FrameEval(clip, functools.partial(
		ProcessFrame,
		clip=clip,
		lcrop=static_crop_left,
		rcrop=static_crop_right,
		tcrop=static_crop_top,
		bcrop=static_crop_bottom,
		mb=MergeBorders,
		fix_horiz=fix_horiz,
		fix_vert=fix_vert
	), prop_src=detect)

	return clip


def PrepOutput(clip):
	args = {'left': crop_left_rem,
		'right': crop_right_rem,
		'format': vs.YUV420P10}
	if preview:
		args.update({'width': clip.width,
			     'height': scaled_height})

	return awf.zresize(clip, **args)


def DoCrop(clip):
	return std.Crop(clip, left=crop_left2, right=crop_right2)


def AlignToCrop(clip):
	# Adds black borders to align the original video with the cropped version when both are centered in the screen.
	return std.AddBorders(
		clip,
		left=max(0, crop_right2-crop_left2),
		right=max(0, crop_left2-crop_right2)
	)


src = bs.VideoSource(fn, cachepath=os.path.expanduser("~/.bsindex.json"))
if preview:
	#uncropped = PrepOutput(AlignToCrop(RestoreProgressive(src)))
	#uncropped.set_output(9)
	unfiltered = PrepOutput(DoCrop(RestoreProgressive(src)))
	unfiltered.set_output(8)
curr = src

if "S03E01" in fn:
	curr = vsfieldkit.fill_analog_frame_ends(
		curr, bottom_blank_width=48, top_blank_width=0)

curr = vs.core.bifrost.Bifrost(curr, interlaced=True)
curr = vs.core.tcomb.TComb(curr, mode=2)
curr = RestoreProgressive(curr)

# bifrost and vfm don't allow 10bit so, can't upsample the processing space until now.
curr = vsutil.depth(curr, 10)

# deblock comes before any crop so it can see the original alignment
curr = haf.Deblock_QED(curr)

curr = DoCrop(curr)

curr = FixEdges(curr,
		fix_vert=8 if "S03E22" in fn else 0,
		static_crop_left=crop_left_true_rem,
		static_crop_right=crop_right_true_rem)


# fill single-pixel black bars
curr = fb.FillBorders(curr,
	left=crop_left_true_rem if (crop_left_true_rem - crop_left_rem) == 1 else 0,
	right=crop_right_true_rem if (crop_right_true_rem - crop_right_rem) == 1 else 0
)


curr = vs.core.neo_vd.VagueDenoiser(curr, threshold=0.4, planes=[1,2])
curr = vs.core.neo_fft3d.FFT3D(curr, bt=5, sigma=1.9, planes=[1,2])

curr = vsdehalo.smooth_dering(curr, mrad=2, minp=4, mthr=0.2, elast=4)

curr = vs.core.neo_fft3d.FFT3D(curr, bt=5, sigma=0.6, planes=[0])
curr = vs.core.neo_vd.VagueDenoiser(curr, threshold=0.18, planes=[0])
# correct the slight blur from the denoise
curr = vs.core.cas.CAS(curr, sharpness=0.07, planes=[0])

curr = kgf.adaptive_grain(curr, static=False)
filtered = PrepOutput(curr)

encfn = fn.replace("remux/", "encode/")
if preview and os.path.exists(encfn):
	enc = bs.VideoSource(encfn, cachepath=os.path.expanduser("~/.bsindex.json"))
	enc = awf.zresize(enc, width=enc.width, height=scaled_height)
	enc.set_output(0)
	filtered.set_output(1)
else:
	filtered.set_output(0)
