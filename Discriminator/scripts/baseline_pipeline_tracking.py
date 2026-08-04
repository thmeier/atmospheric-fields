"""Small W&B adapter shared by baseline pipeline stages."""

import csv
import os
import re
from contextlib import contextmanager
from pathlib import Path

from omegaconf import OmegaConf

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience only
    load_dotenv = None


class DisabledRun:
    """W&B-compatible no-op used by tests and explicitly disabled pipelines."""

    def __init__(self):
        self.summary = {}
        self.url = None

    def log(self, *_args, **_kwargs):
        pass


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "unnamed"


def parsed_csv_value(value):
    if value == "":
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


class PipelineTracker:
    def __init__(self, cfg, pipeline_id):
        settings = cfg.pipeline.wandb
        self.enabled = bool(settings.get("enabled", True))
        self.mode = str(settings.get("mode", "online"))
        self.project = str(settings.get("project", "weather-discriminator-baselines"))
        self.entity = settings.get("entity")
        self.group = str(pipeline_id)
        self.tags = [str(tag) for tag in settings.get("tags", [])]
        self.pipeline_alias = safe_name(pipeline_id)
        self.logged_artifacts = []
        self._wandb = None
        if self.enabled:
            if load_dotenv is not None:
                load_dotenv(Path(__file__).resolve().parents[2] / "wandb_info.env")
            import wandb
            self._wandb = wandb
            if self.mode == "online" and not wandb.login():
                raise RuntimeError("W&B online mode is enabled, but authentication failed.")

    @contextmanager
    def run(self, name, job_type, cfg, metadata=None, tags=None):
        if not self.enabled:
            yield DisabledRun()
            return
        config = OmegaConf.to_container(cfg, resolve=True)
        if metadata:
            config = {"configuration": config, "stage": dict(metadata)}
        run = self._wandb.init(
            project=self.project,
            entity=None if self.entity is None else str(self.entity),
            group=self.group,
            job_type=str(job_type),
            name=str(name),
            tags=self.tags + [str(tag) for tag in (tags or [])],
            config=config,
            mode=self.mode,
            save_code=True,
            reinit="create_new",
        )
        try:
            run.summary["pipeline_id"] = self.group
            run.summary["status"] = "running"
            yield run
        except BaseException as error:
            run.summary["status"] = "failed"
            run.summary["error"] = f"{type(error).__name__}: {error}"
            run.finish(exit_code=1)
            raise
        else:
            run.summary["status"] = "completed"
            run.finish()

    def log_artifact(self, run, name, artifact_type, paths, metadata=None):
        paths = [Path(path) for path in paths if Path(path).is_file()]
        if not self.enabled or not paths:
            return None
        artifact = self._wandb.Artifact(
            safe_name(name), type=str(artifact_type), metadata=dict(metadata or {})
        )
        common_root = Path(Path(paths[0]).anchor)
        try:
            common_root = Path(os.path.commonpath([str(path.parent) for path in paths]))
        except ValueError:
            pass
        for path in paths:
            try:
                artifact_name = str(path.relative_to(common_root))
            except ValueError:
                artifact_name = path.name
            artifact.add_file(str(path), name=artifact_name)
        logged = run.log_artifact(
            artifact, aliases=["latest", self.pipeline_alias]
        )
        self.logged_artifacts.append(logged)
        return logged

    def log_csv_table(self, run, key, path):
        path = Path(path)
        if not self.enabled or not path.is_file():
            return
        with open(path, newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        if rows:
            data = [[parsed_csv_value(value) for value in row] for row in rows[1:]]
            run.log({str(key): self._wandb.Table(columns=rows[0], data=data)})

    def log_images(self, run, paths, root):
        if not self.enabled:
            return
        root = Path(root)
        for path in paths:
            path = Path(path)
            key = "plots/" + str(path.relative_to(root).with_suffix(""))
            run.log({key: self._wandb.Image(str(path))})

    def log_records_table(self, run, key, records):
        if not self.enabled or not records:
            return
        columns = sorted({column for record in records for column in record})
        data = [[record.get(column) for column in columns] for record in records]
        run.log({str(key): self._wandb.Table(columns=columns, data=data)})
