import numpy as np
from grass.script import core
from grass.gunittest.case import TestCase
from grass.gunittest.main import test


class TestIAlbedo(TestCase):
    """Regression tests for the i.albedo GRASS GIS module.
    These tests ensure that the i.albedo module calculates the albedo correctly
    and handles edge cases like zero and one input values."""

    input_rasters = ["modis_band1", "modis_band2", "modis_band3"]
    output_raster = "albedo_output"

    def setUp(self):
        """Setting up the test environment by creating input rasters"""
        self.use_temp_region()
        self.runModule("g.region", n=10, s=0, e=10, w=0, rows=10, cols=10)

        # Set up input rasters
        for i, raster in enumerate(self.input_rasters, 1):
            self.runModule(
                "r.mapcalc", expression=f"{raster} = {i} + row() * 0.1", overwrite=True
            )

    def tearDown(self):
        """Cleaning up the test environment by removing created data"""
        rasters_to_remove = [
            *self.input_rasters,
            self.output_raster,
            "combined",
            "combined_albedo",
        ]
        self.runModule(
            "g.remove",
            type="raster",
            name=",".join(rasters_to_remove),
            flags="f",
            quiet=True,
        )
        self.del_temp_region()

    def checkRaster(self, raster_name):
        """Check if a raster exists in the GRASS database."""
        try:
            core.read_command("r.info", map=raster_name)
            return True
        except Exception:
            return False

    def wait_for_raster(self, raster_name):
        """Wait for the raster to be available before continuing."""
        import time

        start_time = time.time()
        while time.time() - start_time < 60:
            if self.checkRaster(raster_name):
                return
            time.sleep(1)
        self.fail(f"Raster {raster_name} not found after waiting 60 seconds.")

    def test_albedo_output_exists(self):
        """Check that i.albedo produces the expected output raster."""
        self.assertModule(
            "i.albedo",
            input=",".join(self.input_rasters),
            output=self.output_raster,
            overwrite=True,
        )
        self.wait_for_raster(self.output_raster)
        self.assertRasterExists(self.output_raster)

    def test_albedo_output_range(self):
        """Check that the albedo values are within the expected range (0 to 1)."""
        self.assertModule(
            "i.albedo",
            input=",".join(self.input_rasters),
            output=self.output_raster,
            overwrite=True,
        )
        self.wait_for_raster(self.output_raster)

        stats = core.parse_command("r.info", flags="r", map=self.output_raster)
        min_val, max_val = float(stats["min"]), float(stats["max"])
        self.assertGreaterEqual(min_val, 0, "Minimum albedo value is less than 0.")
        self.assertLessEqual(max_val, 1, "Maximum albedo value is greater than 1.")

    def _parse_raster_values(self, raster):
        """Helper function to parse raster values into a numpy array."""
        output = core.read_command(
            "r.stats", input=raster, flags="n", separator="space"
        ).splitlines()
        return np.array(
            [float(line.split()[1]) for line in output if len(line.split()) == 2]
        )

    def test_albedo_output_linearity(self):
        """Test linearity of the albedo calculation by combining input rasters."""
        self.runModule(
            "r.mapcalc",
            expression="combined = modis_band1 + modis_band2",
            overwrite=True,
        )
        self.assertModule(
            "i.albedo",
            input="combined,modis_band3",
            output="combined_albedo",
            overwrite=True,
        )
        self.wait_for_raster("combined_albedo")

        combined_albedo_values = self._parse_raster_values("combined_albedo")
        albedo_values_band1_band2 = self._parse_raster_values(
            "modis_band1"
        ) + self._parse_raster_values("modis_band2")

        self.assertTrue(
            np.allclose(combined_albedo_values, albedo_values_band1_band2, atol=1e-5),
            "Linearity failed for combined albedo calculation",
        )

    def test_noaa_albedo_calculation(self):
        """Test that i.albedo correctly computes albedo using the NOAA AVHRR formula."""

        red_val = 0.5
        nir_val = 0.8

        expected_albedo = 0.035 + 0.545 * nir_val - 0.32 * red_val

        self.runModule(
            "r.mapcalc", expression="noaa_red = {}".format(red_val), overwrite=True
        )
        self.runModule(
            "r.mapcalc", expression="noaa_nir = {}".format(nir_val), overwrite=True
        )

        self.assertModule(
            "i.albedo",
            input="noaa_red,noaa_nir",
            output=self.output_raster,
            overwrite=True,
        )
        self.wait_for_raster(self.output_raster)

        albedo_values = self._parse_raster_values(self.output_raster)

        self.assertTrue(
            np.allclose(albedo_values, expected_albedo, atol=1e-5),
            f"Expected {expected_albedo}, but got {albedo_values}",
        )

    def test_modis_parameter_variation(self):
        """Test MODIS albedo calculation with different flags."""
        flags_combinations = [
            ["-m"],
            ["-m", "-d"],
        ]
        for flags in flags_combinations:
            with self.subTest(flags=flags):
                self.run_albedo(flags, self.test_inputs["modis"])

    def test_all_zero_raster(self):
        """Test i.albedo behavior with an all-zero raster."""
        self.runModule("r.mapcalc", expression="modis_band1 = 0", overwrite=True)
        self.assertModule(
            "i.albedo",
            input="modis_band1,modis_band2,modis_band3",
            output=self.output_raster,
            overwrite=True,
        )
        self.wait_for_raster(self.output_raster)

        albedo_values = self._parse_raster_values(self.output_raster)
        self.assertTrue(
            np.allclose(albedo_values, 0),
            "Albedo should be all zeros for all-zero raster",
        )

    def test_all_one_raster(self):
        """Test i.albedo behavior with an all-one raster."""
        self.runModule("r.mapcalc", expression="modis_band1 = 1", overwrite=True)
        self.assertModule(
            "i.albedo",
            input="modis_band1,modis_band2,modis_band3",
            output=self.output_raster,
            overwrite=True,
        )
        self.wait_for_raster(self.output_raster)

        albedo_values = self._parse_raster_values(self.output_raster)
        self.assertTrue(
            np.all(np.isfinite(albedo_values)), "Albedo should have valid values"
        )

    def test_large_raster_performance(self):
        """Assess performance with a larger raster."""
        self.runModule("g.region", n=90, s=-90, e=180, w=-180, rows=1000, cols=1000)
        self.runModule("r.mapcalc", expression="modis_band1 = col()", overwrite=True)
        self.assertModule(
            "i.albedo",
            input="modis_band1,modis_band2,modis_band3",
            output=self.output_raster,
            overwrite=True,
        )
        self.wait_for_raster(self.output_raster)
        self.assertRasterExists(self.output_raster)


if __name__ == "__main__":
    test()
