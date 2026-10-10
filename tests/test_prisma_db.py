from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.car_scan import prisma_db


class PrismaSubprocessEncodingTests(unittest.TestCase):
    def test_schema_push_decodes_subprocess_output_as_utf8_with_replacement(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with (
            patch.dict("os.environ", {}, clear=False),
            patch.object(prisma_db, "SCHEMA_PATH", prisma_db.Path(__file__)),
            patch.object(prisma_db.subprocess, "run", return_value=completed) as run,
        ):
            prisma_db.push_schema("postgresql://test")

        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(run.call_args.kwargs["errors"], "replace")


if __name__ == "__main__":
    unittest.main()
