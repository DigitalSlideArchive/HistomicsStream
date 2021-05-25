"""Whole-slide image file reader for TensorFlow.

The histomics_stream.codecs module supplies codecs that are useful for Zarr file storage with jpeg or
jpeg2k compression.

"""

from imagecodecs import jpeg_encode, jpeg_decode, jpeg2k_encode, jpeg2k_decode
from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray, ensure_contiguous_ndarray, ndarray_copy
from numcodecs.registry import register_codec


class jpeg(Codec):
    """Codec providing jpeg compression via imagecodecs.

    Parameters
    ----------
    quality : int
        Compression level.

    Notes
    -----
    For the code that uses Zarr data storage for jpeg images, we need to supply codecs.  Note that we
    use this codec instead of that available from the zarr_jpeg package.  The latter collapses
    dimensions by default, can require us to transpose dimensions, and can miss optimizations based upon
    RGB data.

    """

    codec_id = "jpeg"

    def __init__(self, quality=100):
        self.quality = quality
        assert 0 < self.quality <= 100 and isinstance(self.quality, int)
        super().__init__()

    def encode(self, buf):
        """The method to encode a raw image into jpeg format.

        Parameters
        ----------
        buf : ndarray
            The raw image to be encoded into jpeg format

        Returns
        -------
        ndarray
            The image in jpeg format

        """

        bufa = ensure_ndarray(buf)
        assert 2 <= bufa.ndim <= 3
        return jpeg_encode(bufa, level=self.quality)

    def decode(self, buf, out=None):
        """The method to decode a jpeg image into a raw format.

        Parameters
        ----------
        buf : contiguous_ndarray
            The jpeg image to be decoded into raw format.
        out : contiguous_ndarray, optional
            Another location to write the raw image to.

        Returns
        -------
        ndarray
            The image in raw format

        """

        buf = ensure_contiguous_ndarray(buf)
        if out is not None:
            out = ensure_contiguous_ndarray(out)
        tiled = jpeg_decode(buf)
        return ndarray_copy(tiled, out)


register_codec(jpeg)


class jpeg2k(Codec):
    """Codec providing jpeg2k compression via imagecodecs.

    Parameters
    ----------
    quality : int
        Compression level.

    """

    codec_id = "jpeg2k"

    def __init__(self, quality=100):
        self.quality = quality
        assert 0 < self.quality <= 100 and isinstance(self.quality, int)
        super().__init__()

    def encode(self, buf):
        """The method to encode a raw image into jpeg2k format.

        Parameters
        ----------
        buf : ndarray
            The raw image to be encoded into jpeg2k format

        Returns
        -------
        ndarray
            The image in jpeg2k format

        """

        bufa = ensure_ndarray(buf)
        assert 2 <= bufa.ndim <= 3
        return jpeg2k_encode(bufa, level=self.quality)

    def decode(self, buf, out=None):
        """The method to decode a jpeg2k image into a raw format.

        Parameters
        ----------
        buf : contiguous_ndarray
            The jpeg2k image to be decoded into raw format.
        out : contiguous_ndarray, optional
            Another location to write the raw image to.

        Returns
        -------
        ndarray
            The image in raw format

        """

        buf = ensure_contiguous_ndarray(buf)
        if out is not None:
            out = ensure_contiguous_ndarray(out)
        tiled = jpeg2k_decode(buf)
        return ndarray_copy(tiled, out)


register_codec(jpeg2k)
