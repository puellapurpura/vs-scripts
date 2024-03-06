import collections
import functools

import vapoursynth as vs
# do not add any aliases to attributes of vs; holding references can cause
# preview tools to crash on reload.

def CheckPlugin(plugin):
    if not getattr(vs.core, plugin):
        raise Exception("Plugin '%s' not found." % plugin)


def _RoundDown(val, mul):
    return val - (val % mul)


def _RoundUp(val, mul):
    rem = val % mul
    if rem:
        return val + (mul - rem)
    else:
        return val


class Border(collections.namedtuple(
    "BaseBorder", ['left', 'right', 'top', 'bottom'], defaults=[0]*4)):
    def __map(self, other, func):
        if not isinstance(other, Border):
            raise NotImplemented
        return Border(
                left=func(self.left, other.left),
                right=func(self.right, other.right),
                top=func(self.top, other.top),
                bottom=func(self.bottom, other.bottom)
        )

    def __add__(self, other):
        if isinstance(other, int):
            other = Border([other]*4)
        return self.__map(other, lambda x, y: x + y)

    def __sub__(self, other):
        if isinstance(other, int):
            other = Border([other]*4)
        return self.__map(other, lambda x, y: x - y)

    def __bool__(self):
        return bool(self.left or self.right or self.top or self.bottom)

    @property
    def total_width(self):
        return self.left + self.right

    @property
    def total_height(self):
        return self.top + self.bottom


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


def _IfCombed(n, f, comb, prog):
    if f.props['_Combed'] > 0:
        return comb
    else:
        return prog


def DeTelecine(clip, tff=1):
    import vsfieldkit
    clip = vs.core.vivtc.VFM(clip, order=tff)
    deint = vs.core.eedi3m.EEDI3(clip, field=tff)
    clip = vs.core.std.FrameEval(
        clip, functools.partial(_IfCombed, comb=deint, prog=clip),
        prop_src=clip)
    clip = vs.core.vivtc.VDecimate(clip)
    clip = vsfieldkit.resample_as_progressive(clip)
    return clip


def RestoreProgressive(clip, tff=1):
    import vsfieldkit
    clip = vs.core.vivtc.VFM(clip, order=tff)
    deint = vs.core.eedi3m.EEDI3(clip, field=tff)
    clip = vs.core.std.FrameEval(
        clip, functools.partial(_IfCombed, comb=deint, prog=clip),
        prop_src=clip)
    clip = vsfieldkit.resample_as_progressive(clip)
    return clip


def MatchingGray(clip):
    """Returns the grayscale equivalent of clip's format."""
    return getattr(vs, "GRAY%s" % clip.format.bits_per_sample)


def MergeBorders(clip, border_clip, border=None):
    """Merges border of border_clip onto clip."""
    if border is None:
        border = Border()
    Black = functools.partial(
        vs.core.std.BlankClip, format=MatchingGray(clip))
    White = functools.partial(
        Black, color=(2**clip.format.bits_per_sample - 1))
    if border.left > 0 or border.right > 0:
        clips = []
        if border.left > 0:
            clips.append(White(width=border.left, height=clip.height))
        clips.append(Black(
            width=clip.width - border.total_width,
            height=clip.height))
        if border.right > 0:
            clips.append(White(width=border.right, height=clip.height))
        mask = vs.core.std.StackHorizontal(clips)
        clip = vs.core.std.MaskedMerge(clip, border_clip, mask)
    if border.top > 0 or border.bottom > 0:
        clips = []
        if border.top > 0:
            clips.append(White(height=border.top, width=clip.width))
        clips.append(Black(
            height=clip.height - border.total_height,
            width=clip.width))
        if border.bottom > 0:
            clips.append(White(height=border.bottom, width=clip.width))
        mask = vs.core.std.StackVertical(clips)
        clip = vs.core.std.MaskedMerge(clip, border_clip, mask)
    return clip


# mitigate dirty lines around the edges. this is not a perfect solution
# because of how variable these lines can be, but it does a good job of
# reducing their impact.
# Note that this can introduce artifacts where there were none, so it should
# only be used when dirty lines are present in parts of the video, and if lines
# are only horizontal or vertical you should set fix_horiz/vert to 0 to disable
# the direction without lines.
# fix_dist: how far in from detected border of video to fix lines
# fix_horiz: like fix_dist but for left/right only
# fix_vert: like fix_dist but for top/bottom only
# border_detect: how far in to scan for the variable video borders
# static_border: known static black borders to compensate for
def FixEdges(clip, fix_dist=None, border_detect=10,
             fix_horiz=None, fix_vert=None, static_border=Border()):
    CheckPlugin('acrop')
    CheckPlugin('edgefixer')
    CheckPlugin('fb')

    if fix_dist is None:
        fix_dist = 6
    if not any((fix_dist, fix_horiz, fix_vert)):
        return clip
    if fix_horiz is None:
        fix_horiz = fix_dist
    if fix_vert is None:
        fix_vert = fix_dist

    # fill in static borders so any noise won't affect edge detection
    detect = MergeBorders(
        clip,
        vs.core.std.BlankClip(clip),
        static_border
    )
    # exact size of black borders varies shot-by-shot, so we need to detect
    # where the real edge is
    ws = 2**clip.format.subsampling_w
    hs = 2**clip.format.subsampling_h
    detect = vs.core.acrop.CropValues(
        detect,
        left=_RoundDown(border_detect+static_border.left, ws),
        right=_RoundDown(border_detect+static_border.right, ws),
        top=_RoundDown(border_detect+static_border.top, hs),
        bottom=_RoundDown(border_detect+static_border.bottom, hs),
    )

    def ProcessFrame(n, f, clip, border, fix_horiz, fix_vert):
        orig = clip
        crop = Border(
                left=max(border.left, f.props['CropLeftValue']),
                right=max(border.right, f.props['CropRightValue']),
                top=max(border.top, f.props['CropTopValue']),
                bottom=max(border.bottom, f.props['CropBottomValue'])
        )

        # fill in borders so the black areas don't dim the blur
        refbase = vs.core.fb.FillBorders(
            clip, mode="mirror", **crop._asdict())

        # do horiz and vert as separate passes to minimize blurring
        # in the wrong direction.
        if fix_horiz:
            ref = vs.core.std.BoxBlur(refbase, hradius=1, hpasses=1, vpasses=0)
            clip = vs.core.edgefixer.Reference(
                clip, ref,
                left=fix_horiz+crop.left, right=fix_horiz+crop.right)
        if fix_vert:
            ref = vs.core.std.BoxBlur(refbase, vradius=1, hpasses=0, vpasses=1)
            clip = vs.core.edgefixer.Reference(
                clip, ref,
                top=fix_vert+crop.top, bottom=fix_vert+crop.bottom)

        # restore original borders
        clip = MergeBorders(clip, orig, crop)
        return clip

    clip = vs.core.std.FrameEval(clip, functools.partial(
        ProcessFrame,
        clip=clip,
        border=static_border,
        fix_horiz=fix_horiz,
        fix_vert=fix_vert
    ), prop_src=detect)

    return clip


def _AddBorders(clip, border, **kwargs):
    """Like vs.core.std.AddBorders, but rounds borders up to meet subsampling reqs.
    Returns a tuple of the clip and the additional border added by rounding."""
    ws = 2**clip.format.subsampling_w
    hs = 2**clip.format.subsampling_h
    orig_border = border
    border = Border(
            left=_RoundUp(border.left, ws),
            right=_RoundUp(border.right, ws),
            top=_RoundUp(border.top, hs),
            bottom=_RoundUp(border.bottom, hs)
    )
    clip = vs.core.std.AddBorders(clip, **border._asdict(), **kwargs)
    return clip, border - orig_border


def SafeCrop(clip, border):
    """Crop border from clip, respecting chroma subsampling limits.
    Returns a tuple of the cropped clip and the remaining cropping."""
    ws = 2**clip.format.subsampling_w
    hs = 2**clip.format.subsampling_h
    clip = vs.core.std.Crop(
            clip,
            left=_RoundDown(border.left, ws),
            right=_RoundDown(border.right, ws),
            top=_RoundDown(border.top, hs),
            bottom=_RoundDown(border.bottom, hs)
    )
    border = Border(
            left=border.left % ws,
            right=border.right % ws,
            top=border.top % hs,
            bottom=border.bottom % hs
    )
    return clip, border

def AlignToSafeCrop(clip, border):
    """Adds black bars to clip such that it and a SafeCrop-ed clip will center on-screen to the same position. Useful for comparing cropped/uncropped in vspreview to ensure you're not cutting anything off."""
    clip, rem = _AddBorders(clip, Border(
        left=max(0, border.right-border.left),
        right=max(0, border.left-border.right),
        top=max(0, border.bottom-border.top),
        bottom=max(0, border.top-border.bottom)
    ))

    return SafeCrop(clip, rem)[0]

def ZresizeCrop(clip, border, scale_width=None, scale_height=None,
                zresize_kwargs=None):
    """Crop border from clip, using zresize to resample chroma.
    Returns a tuple of the cropped clip and the remaining cropping."""
    import awsmfunc

    if zresize_kwargs is None:
        zresize_kwargs = {}
    else:
        zresize_kwargs = zresize_kwargs.copy()
    in_ws = 2**clip.format.subsampling_w
    in_hs = 2**clip.format.subsampling_h

    # While zresize can resample chroma, we still have to respect the chroma
    # subsampling limits of the output format.
    out_fmt = zresize_kwargs.get("format", clip.format)
    if isinstance(out_fmt, vs.PresetVideoFormat):
        out_fmt = vs.core.std.BlankClip(format=out_fmt).format
    out_ws = 2**out_fmt.subsampling_w
    out_hs = 2**out_fmt.subsampling_h
    orig_border = border
    if (clip.width - (border.left + border.right)) % out_ws:
        if not border.right % out_ws:
            border = border._replace(left=_RoundDown(border.left, out_ws))
        else:
            border = border._replace(right=_RoundDown(border.right, out_ws))
    if (clip.height - (border.top + border.bottom)) % out_hs:
        if not border.bottom % out_hs:
            border = border._replace(bottom=_RoundDown(border.bottom, out_hs))
        else:
            border = border._replace(top=_RoundDown(border.top, out_hs))

    zresize_kwargs.update(border._asdict())
    if scale_width or scale_height:
        if not scale_width:
            scale_width = clip.width
        elif not scale_height:
            scale_height = clip.height
        zresize_kwargs.update({'width': scale_width, 'height': scale_height})
    return awsmfunc.zresize(clip, **zresize_kwargs), orig_border - border


def AlignToZresizeCrop(clip, border, scale_width=None, scale_height=None,
                       zresize_kwargs=None):
    """Adds black bars to clip such that it and a ZresizeCrop-ed clip will center on-screen to the same position. Useful for comparing cropped/uncropped in vspreview to ensure you're not cutting anything off."""
    if zresize_kwargs is None:
        zresize_kwargs = {}
    else:
        zresize_kwargs = zresize_kwargs.copy()
    
    ws = 2**clip.format.subsampling_w
    hs = 2**clip.format.subsampling_h
    
    clip, rem = _AddBorders(clip, Border(
        left=max(0, border.right-border.left),
        right=max(0, border.left-border.right),
        top=max(0, border.bottom-border.top),
        bottom=max(0, border.top-border.bottom)
    ))

    zresize_kwargs.update(rem._asdict())
    if scale_width or scale_height:
        if scale_height:
            scale_height += rem.total_height
        if scale_width:
            scale_width += rem.total_width
        if not scale_width:
            scale_width = clip.width
        elif not scale_height:
            scale_height = clip.height
    return ZresizeCrop(clip, rem,
                       scale_width=scale_width, scale_height=scale_height,
                       zresize_kwargs=zresize_kwargs)[0]
