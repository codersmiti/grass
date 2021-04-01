"""
Created on Mon Jun 11 18:02:27 2012

@author: pietro
"""
import ctypes
import grass.lib.gis as libgis
import grass.lib.raster as libraster
import grass.lib.segment as libseg
from grass.pygrass.raster.raster_type import TYPE as RTYPE


class Segment(object):
    def __init__(self, srows=64, scols=64, maxmem=100):
        self.srows = srows
        self.scols = scols
        self.maxmem = maxmem
        self.c_seg = ctypes.pointer(libseg.SEGMENT())

    def rows(self):
        return libraster.Rast_window_rows()

    def cols(self):
        return libraster.Rast_window_cols()

    def nseg(self):
        rows = self.rows()
        cols = self.cols()
        return int(
            ((rows + self.srows - 1) / self.srows)
            * ((cols + self.scols - 1) / self.scols)
        )

    def segments_in_mem(self):
        if self.maxmem > 0 and self.maxmem < 100:
            seg_in_mem = (self.maxmem * self.nseg()) / 100
        else:
            seg_in_mem = 4 * (self.rows() / self.srows + self.cols() / self.scols + 2)
        if seg_in_mem == 0:
            seg_in_mem = 1
        return seg_in_mem

    def open(self, mapobj):
        """Open a segment it is necessary to pass a RasterSegment object."""
        self.val = RTYPE[mapobj.mtype]["grass def"]()
        size = ctypes.sizeof(RTYPE[mapobj.mtype]["ctypes"])
        file_name = libgis.G_tempfile()
        libseg.Segment_open(
            self.c_seg,
            file_name,
            self.rows(),
            self.cols(),
            self.srows,
            self.scols,
            size,
            self.nseg(),
        )
        self.flush()

    def format(self, mapobj, file_name="", fill=True):
        """The segmentation routines require a disk file to be used for paging
        segments in and out of memory. This routine formats the file open for
        write on file descriptor fd for use as a segment file.
        """
        if file_name == "":
            file_name = libgis.G_tempfile()
        mapobj.temp_file = open(file_name, "w")
        size = ctypes.sizeof(RTYPE[mapobj.mtype]["ctypes"])
        if fill:
            libseg.Segment_format(
                mapobj.temp_file.fileno(),
                self.rows(),
                self.cols(),
                self.srows,
                self.scols,
                size,
            )
        else:
            libseg.Segment_format_nofill(
                mapobj.temp_file.fileno(),
                self.rows(),
                self.cols(),
                self.srows,
                self.scols,
                size,
            )
        # TODO: why should I close and then re-open it?
        mapobj.temp_file.close()

    def init(self, mapobj, file_name=""):
        if file_name == "":
            file_name = mapobj.temp_file.name
        mapobj.temp_file = open(file_name, "w")
        libseg.Segment_init(self.c_seg, mapobj.temp_file.fileno(), self.segments_in_mem)

    def get_row(self, row_index, buf):
        """Return the row using, the `segment` method"""
        libseg.Segment_get_row(self.c_seg, buf.p, row_index)
        return buf

    def put_row(self, row_index, buf):
        """Write the row using the `segment` method"""
        libseg.Segment_put_row(self.c_seg, buf.p, row_index)

    def get(self, row_index, col_index):
        """Return the value of the map"""
        libseg.Segment_get(self.c_seg, ctypes.byref(self.val), row_index, col_index)
        return self.val.value

    def put(self, row_index, col_index):
        """Write the value to the map"""
        libseg.Segment_put(self.c_seg, ctypes.byref(self.val), row_index, col_index)

    def get_seg_number(self, row_index, col_index):
        """Return the segment number"""
        return row_index / self.srows * self.cols / self.scols + col_index / self.scols

    def flush(self):
        """Flush pending updates to disk.
        Forces all pending updates generated by Segment_put() to be written to
        the segment file seg. Must be called after the final Segment_put()
        to force all pending updates to disk. Must also be called before the
        first call to Segment_get_row."""
        libseg.Segment_flush(self.c_seg)

    def close(self):
        """Free memory allocated to segment and delete temp file.  """
        libseg.Segment_close(self.c_seg)

    def release(self):
        """Free memory allocated to segment.
        Releases the allocated memory associated with the segment file seg.
        Note: Does not close the file. Does not flush the data which may be
        pending from previous Segment_put() calls."""
        libseg.Segment_release(self.c_seg)