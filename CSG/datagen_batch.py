import argparse
import os
import trimesh
import numpy as np
import traceback
import time
import re
import gc

import vtk
from vtk.util import numpy_support
from maps import MAPS
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm


def read_vtk_polydata(file_path, retries=3, retry_delay=0.25):
    file_path = str(file_path)
    for attempt in range(retries):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(file_path)
        reader.Update()
        poly_data = reader.GetOutput()
        points_obj = poly_data.GetPoints() if poly_data is not None else None
        points_data = points_obj.GetData() if points_obj is not None else None
        if points_data is not None:
            break
        if attempt + 1 < retries:
            time.sleep(retry_delay)
    else:
        raise ValueError(f"VTK reader returned no points for {file_path}")

    points = numpy_support.vtk_to_numpy(points_data).astype(np.float64)

    polys = poly_data.GetPolys()
    if polys is None:
        raise ValueError(f"VTK reader returned no polygons for {file_path}")
    polys.InitTraversal()
    id_list = vtk.vtkIdList()
    faces = []
    for _ in range(polys.GetNumberOfCells()):
        polys.GetNextCell(id_list)
        faces.append([id_list.GetId(j) for j in range(id_list.GetNumberOfIds())])

    scalars = {}
    point_data = poly_data.GetPointData()
    for i in range(point_data.GetNumberOfArrays()):
        name = point_data.GetArrayName(i)
        if name is None:
            continue
        scalars[name] = numpy_support.vtk_to_numpy(point_data.GetArray(i)).copy()

    return points, np.asarray(faces, dtype=np.int64), scalars


def load_pable_ids(pable_ids=None, pable_vtk=None, pable_scalar="P_able_1200",
                   pable_ids_path=None, pable_ids_out=None):
    if pable_ids is not None:
        ids = np.asarray(pable_ids, dtype=np.int64).reshape(-1)
    elif pable_ids_path is not None:
        ids = np.loadtxt(pable_ids_path, dtype=np.int64).reshape(-1)
    elif pable_vtk is not None:
        _, _, scalars = read_vtk_polydata(pable_vtk)
        if pable_scalar not in scalars:
            available = ", ".join(sorted(scalars.keys()))
            raise KeyError(f"{pable_scalar!r} not found in {pable_vtk}. Available: {available}")
        ids = np.flatnonzero(scalars[pable_scalar] > 0)
    else:
        return None

    ids = np.unique(ids.astype(np.int64))
    if pable_ids_out is not None:
        Path(pable_ids_out).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(pable_ids_out, ids, fmt="%d")
    return ids


def iter_source_meshes(src_root, src_pattern='*.vtk', src_recursive=False):
    root = Path(src_root)
    patterns = [src_pattern] if isinstance(src_pattern, (str, Path)) else list(src_pattern)
    paths = []
    seen = set()
    for pattern in patterns:
        iterator = root.rglob(str(pattern)) if src_recursive else root.glob(str(pattern))
        for path in iterator:
            if not path.is_file():
                continue
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return sorted(paths)


def output_path_for_source(dst_root, src_root, obj_path, variation, preserve_source_tree=False):
    suffix = f'_v{variation}' if variation is not None else ''
    if preserve_source_tree:
        rel = obj_path.relative_to(src_root).with_suffix('')
        return Path(dst_root) / rel.parent / f'{rel.name}{suffix}.obj'
    return Path(dst_root) / f'{obj_path.stem}{suffix}.obj'


NONFINITE_ASCII_TOKEN = re.compile(
    rb'(?<![A-Za-z0-9_+\-.])[-+]?(?:inf|nan)(?![A-Za-z0-9_+\-.])',
    re.IGNORECASE,
)


def vtk_has_nonfinite_ascii_token(path):
    path = Path(path)
    try:
        with path.open('rb') as f:
            header = f.read(512).upper()
            if b'ASCII' not in header:
                return False
            tail = b''
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                window = tail + chunk
                if NONFINITE_ASCII_TOKEN.search(window) is not None:
                    return True
                tail = window[-32:]
    except OSError:
        return False
    return False


def build_parser():
    ap = argparse.ArgumentParser(
        description="MAPS batch mapping: original VTK meshes -> mapped subdivided OBJ meshes")
    ap.add_argument('--src-root', required=True,
                    help='source root containing the original VTK meshes')
    ap.add_argument('--dst-root', required=True,
                    help='output root for the mapped OBJ meshes')
    ap.add_argument('--pable-vtk', required=True,
                    help="P_able template VTK providing the anchor points")
    ap.add_argument('--reject-log-path', required=True,
                    help='log file listing VTKs rejected by the inf/nan ASCII precheck')
    ap.add_argument('--failure-log-path', required=True,
                    help='TSV log of failed meshes (source, output, elapsed, reason)')
    ap.add_argument('--log-path', required=True,
                    help='log file listing the output paths of failed meshes')
    ap.add_argument('--base-size', type=int, default=1024,
                    help='target base mesh size before subdivision')
    ap.add_argument('--max-base-size', type=int, default=1024,
                    help='reject meshes whose actual base mesh exceeds this size')
    ap.add_argument('--depth', type=int, default=3,
                    help='subdivision depth (patch size = 4**depth)')
    ap.add_argument('--src-pattern', default='*.vtk',
                    help='source file glob under --src-root')
    ap.add_argument('--src-recursive', action='store_true',
                    help='search --src-root recursively')
    ap.add_argument('--preserve-source-tree', action=argparse.BooleanOptionalAction, default=None,
                    help='mirror the source directory tree in --dst-root (default: auto, on iff --src-recursive)')
    ap.add_argument('--n-variation', type=int, default=1,
                    help='number of mapping variations per source mesh')
    ap.add_argument('--n-worker', type=int, default=8,
                    help='worker process count (<=0 runs in-process)')
    ap.add_argument('--worker-maxtasksperchild', type=int, default=1,
                    help='Pool maxtasksperchild (<=0 disables restarts)')
    ap.add_argument('--timeout', type=float, default=10000,
                    help='per-mesh MAPS timeout in seconds')
    ap.add_argument('--hard-timeout', type=float, default=14400,
                    help='hard per-job timeout in seconds')
    ap.add_argument('--verbose', action='store_true',
                    help='print worker tracebacks')
    ap.add_argument('--reject-nonfinite-vtk', action=argparse.BooleanOptionalAction, default=True,
                    help='precheck VTKs for inf/nan ASCII tokens and skip offenders')
    ap.add_argument('--append-failure-logs', action=argparse.BooleanOptionalAction, default=True,
                    help='append to (vs recreate) the failure/reject logs')
    ap.add_argument('--pable-scalar', default='P_able_1200',
                    help='point-data array name in --pable-vtk marking the anchor points')
    ap.add_argument('--pable-ids-path', default=None,
                    help='optional text file of anchor vertex ids (overrides --pable-vtk)')
    ap.add_argument('--pable-ids-out', default=None,
                    help='optional output file for the resolved anchor ids')
    ap.add_argument('--anatomy-weight', type=float, default=1.0,
                    help='anatomy term weight in the MAPS objective')
    ap.add_argument('--protect-pable', action=argparse.BooleanOptionalAction, default=False,
                    help='protect anchor points during simplification')
    ap.add_argument('--soft-last-pable', action=argparse.BooleanOptionalAction, default=True,
                    help='soft anchor protection at the last simplification level')
    return ap


def maps_async(obj_path, out_path, base_size, max_base_size, depth, timeout,
               trial=1, verbose=False, pable_ids=None, anatomy_weight=1.0,
               protect_pable=True, soft_last_pable=False):
    last_reason = 'unknown'
    last_elapsed = 0.0
    for _ in range(trial):
        start_time = time.time()
        try:
            points, faces, _ = read_vtk_polydata(obj_path)
            maps = MAPS(
                np.array(points), np.array(faces), base_size,
                timeout=timeout, verbose=verbose, pable_ids=pable_ids,
                anatomy_weight=anatomy_weight, protect_pable=protect_pable,
                soft_last_pable=soft_last_pable,
            )
            if maps.base_size > max_base_size:
                last_elapsed = time.time() - start_time
                reason_prefix = 'base_too_large'
                if timeout is not None and last_elapsed >= timeout * 0.95:
                    reason_prefix = 'timeout_or_base_too_large'
                last_reason = (
                    f'{reason_prefix}: actual_base_size={maps.base_size} '
                    f'max_base_size={max_base_size} elapsed={last_elapsed:.1f}s'
                )
                continue
            sub_mesh = maps.mesh_upsampling(depth=depth)
            sub_mesh.export(out_path)
            last_elapsed = time.time() - start_time
            return True, out_path, 'success', last_elapsed, obj_path
        except Exception as e:
            last_elapsed = time.time() - start_time
            last_reason = f'{type(e).__name__}: {e}'
            if verbose:
                traceback.print_exc()
        finally:
            gc.collect()

    return False, out_path, last_reason, last_elapsed, obj_path


def make_MAPS_dataset(dst_root, src_root, base_size, depth, n_variation=None,
                      n_worker=1, timeout=None, max_base_size=None,
                      src_pattern='*.vtk', src_recursive=False,
                      preserve_source_tree=None, verbose=False,
                      reject_nonfinite_vtk=True, reject_log_path='./rejected_vtk.log',
                      failure_log_path='./maps_failures.tsv', append_failure_logs=True,
                      log_path='maps.log',
                      worker_maxtasksperchild=None, hard_timeout=None,
                      pable_ids=None, pable_vtk=None, pable_scalar='P_able_1200',
                      pable_ids_path=None, pable_ids_out=None,
                      anatomy_weight=1.0, protect_pable=True, soft_last_pable=False):
    if max_base_size is None:
        max_base_size = base_size
    if n_variation is None:
        n_variation = 1
    if preserve_source_tree is None:
        preserve_source_tree = bool(src_recursive)

    pable_ids = load_pable_ids(
        pable_ids=pable_ids, pable_vtk=pable_vtk, pable_scalar=pable_scalar,
        pable_ids_path=pable_ids_path, pable_ids_out=pable_ids_out,
    )

    Path(dst_root).mkdir(parents=True, exist_ok=True)
    append_failure_logs = bool(append_failure_logs)

    log_path = Path(log_path).resolve()
    if log_path.exists() and not append_failure_logs:
        log_path.unlink()

    success_count = 0
    fail_count = 0
    skipped_count = 0

    fail_detail_path = Path(failure_log_path).resolve() if failure_log_path else None
    if fail_detail_path is not None:
        fail_detail_path.parent.mkdir(parents=True, exist_ok=True)
        if fail_detail_path.exists() and not append_failure_logs:
            fail_detail_path.unlink()
        if not fail_detail_path.exists() or fail_detail_path.stat().st_size == 0:
            fail_detail_path.write_text('source\toutput\telapsed_seconds\treason\n', encoding='utf-8')

    def clean_field(value):
        return str(value).replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')

    def record_result(success, path, reason='', elapsed_seconds=0.0, source_path=''):
        nonlocal success_count, fail_count
        if success:
            success_count += 1
            return
        fail_count += 1
        with open(log_path, 'a') as f:
            f.write(str(path) + '\n')
        if fail_detail_path is not None:
            with fail_detail_path.open('a', encoding='utf-8') as f:
                f.write(
                    f'{clean_field(source_path)}\t{clean_field(path)}\t'
                    f'{float(elapsed_seconds):.3f}\t{clean_field(reason)}\n'
                )

    obj_paths = iter_source_meshes(src_root, src_pattern, src_recursive)
    print(f'[SOURCE] found {len(obj_paths)} mesh files under {src_root}')

    rejected_paths = []
    if reject_nonfinite_vtk:
        kept_paths = []
        for obj_path in tqdm(obj_paths, desc='precheck finite VTK', unit='file'):
            if vtk_has_nonfinite_ascii_token(obj_path):
                rejected_paths.append(obj_path)
            else:
                kept_paths.append(obj_path)
        obj_paths = kept_paths
        if rejected_paths:
            reject_log = Path(reject_log_path).resolve()
            reject_log.parent.mkdir(parents=True, exist_ok=True)
            reject_log.write_text(''.join(f'{path}\n' for path in rejected_paths), encoding='utf-8')
            print(f'[REJECT] skipped {len(rejected_paths)} VTK files with inf/nan tokens')

    pbar = tqdm(total=len(obj_paths) * n_variation)

    if worker_maxtasksperchild is not None and worker_maxtasksperchild <= 0:
        worker_maxtasksperchild = None
    if hard_timeout is None and timeout is not None:
        hard_timeout = timeout + max(600.0, float(timeout) * 0.25)
    if hard_timeout is not None and hard_timeout <= 0:
        hard_timeout = None

    jobs = []
    for obj_path in obj_paths:
        for var in range(n_variation):
            dst_path = output_path_for_source(
                dst_root, src_root, obj_path,
                var if n_variation > 1 else None, preserve_source_tree,
            )
            if dst_path.exists():
                skipped_count += 1
                pbar.update()
                continue
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            jobs.append({
                'source': str(obj_path),
                'output': str(dst_path),
                'args': (str(obj_path), str(dst_path), base_size, max_base_size,
                         depth, timeout, 1, verbose, pable_ids,
                         anatomy_weight, protect_pable, soft_last_pable),
            })

    def result_from_worker_error(error, job):
        return (False, job['output'],
                f'worker_error: {type(error).__name__}: {error}',
                0.0, job['source'])

    try:
        if n_worker > 0 and jobs:
            pool = Pool(processes=n_worker, maxtasksperchild=worker_maxtasksperchild)
            in_flight = {}
            next_job = 0
            pool_terminated = False
            try:
                while next_job < len(jobs) or in_flight:
                    while next_job < len(jobs) and len(in_flight) < n_worker:
                        job = jobs[next_job]
                        ret = pool.apply_async(maps_async, job['args'])
                        in_flight[ret] = {'job': job, 'started_at': time.time()}
                        next_job += 1

                    completed = []
                    for ret, meta in list(in_flight.items()):
                        if not ret.ready():
                            continue
                        job = meta['job']
                        try:
                            result = ret.get(timeout=0)
                        except Exception as e:
                            result = result_from_worker_error(e, job)
                        record_result(*result)
                        pbar.update()
                        completed.append(ret)
                    for ret in completed:
                        in_flight.pop(ret, None)

                    if hard_timeout is not None:
                        now = time.time()
                        timed_out = [(ret, meta) for ret, meta in in_flight.items()
                                     if now - meta['started_at'] > hard_timeout]
                        if timed_out:
                            for ret, meta in timed_out:
                                job = meta['job']
                                elapsed = now - meta['started_at']
                                record_result(False, job['output'],
                                              f'hard_timeout: exceeded {hard_timeout:.1f}s',
                                              elapsed, job['source'])
                                pbar.update()
                                in_flight.pop(ret, None)
                            pool.terminate()
                            pool_terminated = True
                            raise TimeoutError(f'hard_timeout exceeded for {len(timed_out)} job(s)')

                    if in_flight and not completed:
                        time.sleep(1.0)
                pool.close()
            except Exception:
                if not pool_terminated:
                    pool.terminate()
                raise
            finally:
                pool.join()
        elif n_worker <= 0:
            for job in jobs:
                try:
                    result = maps_async(*job['args'])
                except Exception as e:
                    result = result_from_worker_error(e, job)
                record_result(*result)
                pbar.update()
    finally:
        pbar.close()

    total_jobs = len(obj_paths) * n_variation
    rejected_count = len(rejected_paths) * n_variation
    print(f'[SUMMARY] total={total_jobs} success={success_count} '
          f'failed={fail_count} skipped={skipped_count} rejected={rejected_count}')


if __name__ == "__main__":
    args = build_parser().parse_args()
    make_MAPS_dataset(
        args.dst_root, args.src_root, args.base_size, args.depth,
        n_variation=args.n_variation, n_worker=args.n_worker,
        timeout=args.timeout, max_base_size=args.max_base_size,
        src_pattern=args.src_pattern,
        src_recursive=args.src_recursive,
        preserve_source_tree=args.preserve_source_tree,
        verbose=args.verbose,
        reject_nonfinite_vtk=args.reject_nonfinite_vtk,
        reject_log_path=args.reject_log_path,
        failure_log_path=args.failure_log_path,
        log_path=args.log_path,
        append_failure_logs=args.append_failure_logs,
        worker_maxtasksperchild=args.worker_maxtasksperchild,
        hard_timeout=args.hard_timeout,
        pable_vtk=args.pable_vtk,
        pable_scalar=args.pable_scalar,
        pable_ids_path=args.pable_ids_path,
        pable_ids_out=args.pable_ids_out,
        anatomy_weight=args.anatomy_weight,
        protect_pable=args.protect_pable,
        soft_last_pable=args.soft_last_pable,
    )
