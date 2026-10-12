# -*- coding: utf-8 -*-
"""Composer attachment tests. No live Dify, no network, no real MAT files."""
import unittest

from lovis_attachments import (
    MAT_MAX_BYTES, CSV_MAX_BYTES, CSV_MAX_ROWS, CSV_FIELDS,
    attachment_meta, source_rate_hint, validate_trend_csv,
)

SAMPLE=(
    "time,current_a,current_b,current_c,bearing_temp\n"
    "2026-10-10 08:00,10,11,10,51\n"
    "2026-10-10 09:00,10,11,11,52\n"
    "2026-10-10 10:00,11,12,11,54\n"
    "2026-10-10 11:00,11,12,11,58\n"
)

class ComposerAttachmentTests(unittest.TestCase):
    def test_valid_csv_for_trend(self):
        text,count=validate_trend_csv(SAMPLE.encode("utf-8"))
        self.assertEqual(count,4)
        self.assertEqual(text.splitlines()[0],",".join(CSV_FIELDS))
        data=attachment_meta("motor_data.csv",SAMPLE.encode("utf-8"))
        self.assertEqual(data["rows"],4)
        self.assertEqual(data["kind"],"csv")
        self.assertNotIn("path",str(data))

    def test_bom_and_optional_blank_measurements(self):
        lines=SAMPLE.replace("10,11,10,51","10,,10,51")
        cleaned,count=validate_trend_csv(lines.encode("utf-8-sig"))
        self.assertIn(",10,,10,51",cleaned)
        self.assertEqual(count,4)

    def test_reject_wrong_header_and_timestamp(self):
        with self.assertRaises(ValueError):
            validate_trend_csv(b"a,b,c\n1,2,3")
        with self.assertRaises(ValueError):
            validate_trend_csv(SAMPLE.replace("2026-10-10 09:00",
                                            "2026-10-10 07:00").encode())
        with self.assertRaises(ValueError):
            validate_trend_csv(SAMPLE.replace("2026-10-10 08:00",
                                            "not-a-time").encode())

    def test_reject_code_in_csv_and_nonfinite_values(self):
        for bad in ('=SYSTEM()','NaN','inf','-4','1e99999'):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    validate_trend_csv(SAMPLE.replace("10,11,10,51",
                                                     bad+",11,10,51").encode())

    def test_limit_and_extension(self):
        with self.assertRaises(ValueError):
            attachment_meta("foo.exe",b"bad")
        with self.assertRaises(ValueError):
            attachment_meta("foo.csv",b"x"* (CSV_MAX_BYTES+1))
        with self.assertRaises(ValueError):
            attachment_meta("foo.mat",b"x"* (MAT_MAX_BYTES+1))
        entries=["2026-10-10 08:00,10,11,10,51"]
        for i in range(1,CSV_MAX_ROWS+1):
            entries.append(f"2026-10-10 08:{i//60:02d}:{i%60:02d},10,11,10,51")
        with self.assertRaises(ValueError):
            validate_trend_csv((",".join(CSV_FIELDS)+"\n"+"\n".join(entries)).encode())

    def test_cwru_sampling_rate_hint_is_only_a_hint(self):
        self.assertEqual(source_rate_hint("98.mat"),48000)
        self.assertEqual(source_rate_hint("100.MAT"),48000)
        self.assertEqual(source_rate_hint("105.mat"),12000)
        self.assertEqual(source_rate_hint("motor.mat"),12000)

if __name__=="__main__":
    unittest.main()
