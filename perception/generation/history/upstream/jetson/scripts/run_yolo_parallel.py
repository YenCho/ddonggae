import argparse
import concurrent.futures
from collections import deque
import subprocess
import threading
import time
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tqdm import tqdm

from run_yolo_coco_composite_batch import write_data_yaml


SEOUL_TZ = ZoneInfo("Asia/Seoul")


def existing_count(output):
    image_dir = Path(output) / "images" / "train"
    if not image_dir.exists():
        return 0
    return len(list(image_dir.glob("*.jpg")))


def image_done(output, image_id, require_meta=False):
    output = Path(output)
    stem = f"{image_id:06d}"
    done = (
        (output / "images" / "train" / f"{stem}.jpg").exists()
        and (output / "labels" / "train" / f"{stem}.txt").exists()
    )
    if require_meta:
        done = done and (output / "_meta" / "train" / f"{stem}.json").exists()
    return done


def make_id_chunks(args):
    if args.resume:
        ids = [
            image_id
            for image_id in range(args.num_images)
            if not image_done(args.output, image_id, require_meta=args.meta_v2)
        ]
    else:
        ids = list(range(args.num_images))

    if args.process_chunk_size > 0:
        size = max(1, int(args.process_chunk_size))
        return [ids[idx:idx + size] for idx in range(0, len(ids), size)]

    chunks = [[] for _ in range(max(1, args.workers))]
    for idx, image_id in enumerate(ids):
        chunks[idx % len(chunks)].append(image_id)
    return [chunk for chunk in chunks if chunk]


def write_id_file(output, worker_idx, ids, attempt=0):
    chunk_dir = Path(output) / "_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    path = chunk_dir / f"worker_{worker_idx:04d}_try_{attempt:02d}.txt"
    path.write_text("\n".join(str(image_id) for image_id in ids) + "\n", encoding="utf-8")
    return path


def run_chunk(args, worker_idx, ids, attempt=0):
    if args.worker_start_delay > 0:
        time.sleep(args.worker_start_delay * worker_idx)
    id_file = write_id_file(args.output, worker_idx, ids, attempt)
    log_dir = Path(args.output) / "_chunks" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"worker_{worker_idx:04d}_try_{attempt:02d}.out.log"
    stderr_path = log_dir / f"worker_{worker_idx:04d}_try_{attempt:02d}.err.log"
    cmd = [
        "blenderproc", "run", args.script,
        "--asset_dir", args.asset_dir,
        "--background_dir", args.background_dir,
        "--arena_background_ratio", str(args.arena_background_ratio),
        "--arena_floor_material", args.arena_floor_material,
        "--arena_wall_material", args.arena_wall_material,
        "--arena_full_background_ratio", str(args.arena_full_background_ratio),
        "--wall_contact_ratio", str(args.wall_contact_ratio),
        "--corner_scene_ratio", str(args.corner_scene_ratio),
        "--motion_blur_hard_negative_ratio", str(args.motion_blur_hard_negative_ratio),
        "--plain_cube_hard_negative_ratio", str(args.plain_cube_hard_negative_ratio),
        "--fruit_texture_dir", args.fruit_texture_dir,
        "--output", args.output,
        "--id_file", str(id_file),
        "--width", str(args.width),
        "--height", str(args.height),
        "--samples", str(args.samples),
        "--cpu_threads", str(args.cpu_threads),
        "--seed", str(args.seed),
        "--min_objects", str(args.min_objects),
        "--max_objects", str(args.max_objects),
        "--scale_min", str(args.scale_min),
        "--scale_max", str(args.scale_max),
        "--max_covered_ratio", str(args.max_covered_ratio),
        "--min_object_visible_ratio", str(args.min_object_visible_ratio),
        "--min_projected_area", str(args.min_projected_area),
        "--min_fruit_visible_ratio", str(args.min_fruit_visible_ratio),
        "--min_fruit_face_pixels", str(args.min_fruit_face_pixels),
        "--single_object_scale_min", str(args.single_object_scale_min),
        "--single_object_min_projected_area", str(args.single_object_min_projected_area),
        "--single_object_min_fruit_face_pixels", str(args.single_object_min_fruit_face_pixels),
        "--min_fruit_face_side", str(args.min_fruit_face_side),
        "--label_format", args.label_format,
        "--min_segment_area", str(args.min_segment_area),
        "--seg_contour_mode", args.seg_contour_mode,
        "--seg_contour_epsilon_ratio", str(args.seg_contour_epsilon_ratio),
        "--negative_ratio", str(args.negative_ratio),
        "--fruit_visibility_tier", args.fruit_visibility_tier,
        "--fruit_visibility_easy_weight", str(args.fruit_visibility_easy_weight),
        "--fruit_visibility_mid_weight", str(args.fruit_visibility_mid_weight),
        "--fruit_visibility_hard_weight", str(args.fruit_visibility_hard_weight),
        "--fruit_class_weight_scale", str(args.fruit_class_weight_scale),
        "--hard_min_fruit_visible_ratio", str(args.hard_min_fruit_visible_ratio),
        "--hard_min_fruit_face_pixels", str(args.hard_min_fruit_face_pixels),
        "--lighting_mode", args.lighting_mode,
        "--fruit_texture_aug", args.fruit_texture_aug,
        "--fruit_texture_layout", args.fruit_texture_layout,
        "--fruit_texture_collage_prob", str(args.fruit_texture_collage_prob),
        "--lens_distortion_prob", str(args.lens_distortion_prob),
        "--render_device", args.render_device,
        "--gpu_device_type", args.gpu_device_type,
        "--ideal_visibility_max_attempts", str(args.ideal_visibility_max_attempts),
    ]
    if args.fruit_only:
        cmd.append("--fruit_only")
    if args.arena_booster_mode:
        cmd.append("--arena_booster_mode")
    if args.robot_camera_view:
        cmd.append("--robot_camera_view")
    if args.apple_orange_boost:
        cmd.append("--apple_orange_boost")
    if args.single_fruit_class_per_image:
        cmd.append("--single_fruit_class_per_image")
    else:
        cmd.append("--allow_mixed_fruit_classes_per_image")
    if args.single_fruit_texture_per_cube:
        cmd.append("--single_fruit_texture_per_cube")
    else:
        cmd.append("--allow_multiple_fruit_textures_per_cube")
    if args.canonical_fruit_textures:
        cmd.append("--canonical_fruit_textures")
    if args.ideal_visibility:
        cmd.append("--ideal_visibility")
    if args.ideal_visibility_debug:
        cmd.append("--ideal_visibility_debug")
    if args.meta_v2:
        cmd.append("--meta_v2")
    if args.keep_generated_face_textures:
        cmd.append("--keep_generated_face_textures")
    if not args.ideal_visibility_match_tier:
        cmd.append("--no_ideal_visibility_match_tier")
    if args.resume:
        cmd.append("--resume")
    with open(stdout_path, "w", encoding="utf-8") as stdout, open(stderr_path, "w", encoding="utf-8") as stderr:
        subprocess.run(cmd, check=True, stdout=stdout, stderr=stderr)
    return worker_idx, len(ids)


def unfinished_ids(args, ids):
    return [
        image_id
        for image_id in ids
        if not image_done(args.output, image_id, require_meta=args.meta_v2)
    ]


def format_eta(seconds):
    if seconds <= 0:
        return "complete"
    minutes = int(round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"about {hours}h {minutes}m"
    return f"about {minutes}m"


def format_seoul_time(timestamp=None):
    if timestamp is None:
        dt = datetime.now(SEOUL_TZ)
    else:
        dt = datetime.fromtimestamp(timestamp, SEOUL_TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S KST")


def format_expected_finish(remaining_seconds):
    if remaining_seconds is None:
        return "calculating"
    return format_seoul_time(time.time() + max(remaining_seconds, 0))


def send_progress_email(args, subject, body):
    if not args.email_to:
        return
    cmd = [
        sys.executable,
        "scripts/send_progress_email.py",
        "--config", args.email_config,
        "--to", args.email_to,
        "--subject", subject,
        "--body", body,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"email notification failed: {exc}", flush=True)


def run_rate(done, initial_done, start_time):
    now = time.time()
    elapsed = max(now - start_time, 1e-6)
    generated = max(done - initial_done, 0)
    return generated / elapsed, generated, elapsed


def build_progress_body(args, done, start_time, initial_done=0):
    now = time.time()
    rate, generated, elapsed = run_rate(done, initial_done, start_time)
    remaining = max(args.num_images - done, 0)
    eta = format_eta(remaining / rate) if rate > 0 else "계산 중"
    return (
        f"YOLO 데이터 생성 진행 상태\n\n"
        f"출력 폴더: {args.output}\n"
        f"진행: {done}/{args.num_images}\n"
        f"이번 실행 생성: {generated}\n"
        f"남은 이미지: {remaining}\n"
        f"이번 실행 평균 속도: {rate:.3f} img/s\n"
        f"remaining time: {eta}\n"
        f"expected finish (Asia/Seoul): {format_expected_finish((remaining / rate) if rate > 0 else None)}\n"
        f"email sent (Asia/Seoul): {format_seoul_time(now)}\n"
        f"elapsed this run: {format_eta(elapsed)}\n"
        f"workers: {args.workers}\n"
        f"samples: {args.samples}\n"
        f"single fruit class per image: {args.single_fruit_class_per_image}\n"
        f"single fruit texture per cube: {args.single_fruit_texture_per_cube}\n"
        f"fruit texture layout: {args.fruit_texture_layout}\n"
        f"texture: {args.fruit_texture_dir}\n"
    )


def build_start_body(args, done, chunks):
    remaining = max(args.num_images - done, 0)
    return (
        "YOLO data generation started.\n\n"
        f"output: {args.output}\n"
        f"progress: {done}/{args.num_images}\n"
        f"remaining: {remaining}\n"
        f"resume mode: {args.resume}\n"
        f"expected finish (Asia/Seoul): calculating after first progress update\n"
        f"email sent (Asia/Seoul): {format_seoul_time()}\n"
        f"workers: {args.workers}\n"
        f"active chunks: {len(chunks)}\n"
        f"process chunk size: {args.process_chunk_size}\n"
        f"max worker retries: {args.max_worker_retries}\n"
        f"render device: {args.render_device}\n"
        f"samples: {args.samples}\n"
        f"image size: {args.width}x{args.height}\n"
        f"objects: {args.min_objects}~{args.max_objects}\n"
        f"label format: {args.label_format}\n"
        f"ideal visibility: {args.ideal_visibility or args.ideal_visibility_debug}\n"
        f"ideal debug: {args.ideal_visibility_debug}\n"
        f"keep generated face textures: {args.keep_generated_face_textures}\n"
        f"single fruit class per image: {args.single_fruit_class_per_image}\n"
        f"single fruit texture per cube: {args.single_fruit_texture_per_cube}\n"
        f"fruit texture layout: {args.fruit_texture_layout}\n"
        f"email every: {args.email_every}\n"
        f"texture: {args.fruit_texture_dir}\n"
    )


def monitor_progress(args, progress, stop_event, start_time, initial_done, interval=5.0):
    last_done = progress.n
    last_time = start_time
    last_email_done = progress.n
    while not stop_event.wait(interval):
        done = min(existing_count(args.output), args.num_images)
        now = time.time()
        delta_done = done - last_done
        delta_time = now - last_time
        instant_rate = delta_done / delta_time if delta_time > 0 and delta_done >= 0 else 0.0
        avg_rate, generated, _ = run_rate(done, initial_done, start_time)
        remaining = max(args.num_images - done, 0)
        eta_min = remaining / avg_rate / 60 if avg_rate > 0 else 0.0

        if done > progress.n:
            progress.update(done - progress.n)
        progress.set_postfix({
            "img/s": f"{avg_rate:.2f}",
            "now": f"{instant_rate:.2f}",
            "run": generated,
            "ETA": f"{eta_min:.1f}m" if avg_rate > 0 else "calc",
        }, refresh=True)

        if args.email_to and args.email_every > 0 and done < args.num_images and done - last_email_done >= args.email_every:
            send_progress_email(
                args,
                f"[진행] YOLO 데이터 생성 {done}/{args.num_images}",
                build_progress_body(args, done, start_time, initial_done),
            )
            last_email_done = done

        last_done = done
        last_time = now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--worker_start_delay",
        type=float,
        default=0.0,
        help="Seconds to stagger each worker launch. Useful when many Blender workers allocate GPU memory at startup.",
    )
    parser.add_argument("--asset_dir", type=str, default="assets/generated")
    parser.add_argument("--background_dir", type=str, default="datasets/backgrounds/coco2017")
    parser.add_argument("--arena_background_ratio", type=float, default=0.0)
    parser.add_argument("--arena_booster_mode", action="store_true")
    parser.add_argument("--arena_floor_material", choices=["sun111_wood"], default="sun111_wood")
    parser.add_argument("--arena_wall_material", choices=["sun168_beige"], default="sun168_beige")
    parser.add_argument("--arena_full_background_ratio", type=float, default=1.0)
    parser.add_argument("--robot_camera_view", action="store_true")
    parser.add_argument("--wall_contact_ratio", type=float, default=0.25)
    parser.add_argument("--corner_scene_ratio", type=float, default=0.10)
    parser.add_argument("--motion_blur_hard_negative_ratio", type=float, default=0.10)
    parser.add_argument("--plain_cube_hard_negative_ratio", type=float, default=0.05)
    parser.add_argument("--apple_orange_boost", action="store_true")
    parser.add_argument("--fruit_texture_dir", type=str, default="datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10")
    parser.add_argument("--output", type=str, default="datasets/yolo_8class_1000")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument("--script", type=str, default="scripts/generate_yolo_coco_composite.py")
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--min_objects", type=int, default=1)
    parser.add_argument("--max_objects", type=int, default=10)
    parser.add_argument("--scale_min", type=float, default=0.40)
    parser.add_argument("--scale_max", type=float, default=2.45)
    parser.add_argument("--max_covered_ratio", type=float, default=0.80)
    parser.add_argument("--min_object_visible_ratio", type=float, default=0.10)
    parser.add_argument("--min_projected_area", type=int, default=900)
    parser.add_argument("--min_fruit_visible_ratio", type=float, default=0.10)
    parser.add_argument("--min_fruit_face_pixels", type=int, default=300)
    parser.add_argument("--single_object_scale_min", type=float, default=0.82)
    parser.add_argument("--single_object_min_projected_area", type=int, default=2600)
    parser.add_argument("--single_object_min_fruit_face_pixels", type=int, default=850)
    parser.add_argument("--min_fruit_face_side", type=int, default=10)
    parser.add_argument("--label_format", choices=["bbox", "segment"], default="bbox")
    parser.add_argument("--min_segment_area", type=int, default=30)
    parser.add_argument("--seg_contour_mode", choices=["all", "largest"], default="largest")
    parser.add_argument("--seg_contour_epsilon_ratio", type=float, default=0.01)
    parser.add_argument("--negative_ratio", type=float, default=0.18)
    parser.add_argument("--fruit_visibility_tier", type=str, default="mixed", choices=["mixed", "easy", "mid", "hard"])
    parser.add_argument("--fruit_visibility_easy_weight", type=float, default=0.50)
    parser.add_argument("--fruit_visibility_mid_weight", type=float, default=0.45)
    parser.add_argument("--fruit_visibility_hard_weight", type=float, default=0.05)
    parser.add_argument("--fruit_class_weight_scale", type=float, default=1.20)
    parser.add_argument("--hard_min_fruit_visible_ratio", type=float, default=-1.0)
    parser.add_argument("--hard_min_fruit_face_pixels", type=int, default=-1)
    parser.add_argument("--lighting_mode", choices=["random", "soft_overhead"], default="random")
    parser.add_argument("--fruit_texture_aug", choices=["none", "light", "strong"], default="strong")
    parser.add_argument("--fruit_texture_layout", choices=["single", "collage", "mixed"], default="single")
    parser.add_argument("--fruit_texture_collage_prob", type=float, default=0.35)
    parser.add_argument("--lens_distortion_prob", type=float, default=0.35)
    parser.add_argument(
        "--render_device",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="Forward render device selection to the generator. Use cpu only as a stability fallback.",
    )
    parser.add_argument(
        "--gpu_device_type",
        type=str,
        default="",
        help="Optional BlenderProc GPU device type preference such as OPTIX or CUDA.",
    )
    parser.add_argument("--canonical_fruit_textures", action="store_true")
    parser.add_argument("--fruit_only", action="store_true")
    parser.add_argument(
        "--single_fruit_class_per_image",
        action="store_true",
        default=False,
        help="Force every fruit cube in one generated image to use the same fruit class.",
    )
    parser.add_argument(
        "--allow_mixed_fruit_classes_per_image",
        dest="single_fruit_class_per_image",
        action="store_false",
        help="Allow different fruit classes to appear in the same generated image.",
    )
    parser.add_argument(
        "--single_fruit_texture_per_cube",
        action="store_true",
        default=False,
        help="Use one source fruit texture on every fruit face of a cube.",
    )
    parser.add_argument(
        "--allow_multiple_fruit_textures_per_cube",
        dest="single_fruit_texture_per_cube",
        action="store_false",
        help="Allow different same-class source texture files on different faces of one cube.",
    )
    parser.add_argument("--ideal_visibility", action="store_true")
    parser.add_argument("--ideal_visibility_debug", action="store_true")
    parser.add_argument("--ideal_visibility_max_attempts", type=int, default=80)
    parser.add_argument("--ideal_visibility_match_tier", action="store_true", default=True)
    parser.add_argument("--no_ideal_visibility_match_tier", dest="ideal_visibility_match_tier", action="store_false")
    parser.add_argument("--meta_v2", action="store_true", help="Forward --meta_v2 to the generator.")
    parser.add_argument(
        "--keep_generated_face_textures",
        action="store_true",
        help="Forward --keep_generated_face_textures to retain generated collage debug textures.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--process_chunk_size",
        type=int,
        default=0,
        help="If >0, split work into many short-lived BlenderProc processes of this many image ids.",
    )
    parser.add_argument("--max_worker_retries", type=int, default=5)
    parser.add_argument("--retry_delay_seconds", type=float, default=30.0)
    parser.add_argument("--email_to", type=str, default="")
    parser.add_argument("--email_every", type=int, default=0)
    parser.add_argument("--email_config", type=str, default="config/email_smtp.json")
    args = parser.parse_args()
    if args.arena_booster_mode:
        args.arena_background_ratio = args.arena_full_background_ratio
        args.robot_camera_view = True if not args.robot_camera_view else args.robot_camera_view
        args.lighting_mode = "soft_overhead" if args.lighting_mode == "random" else args.lighting_mode
        args.apple_orange_boost = True if not args.apple_orange_boost else args.apple_orange_boost

    Path(args.output).mkdir(parents=True, exist_ok=True)
    write_data_yaml(args.output)

    start = time.time()
    done = existing_count(args.output) if args.resume else 0
    initial_done = done
    chunks = make_id_chunks(args)
    if not chunks:
        print(f"[{done}/{args.num_images}] nothing to generate", flush=True)
        if args.email_to:
            send_progress_email(
                args,
                f"[done] YOLO data generation already complete {done}/{args.num_images}",
                build_start_body(args, done, chunks),
            )
        return

    if args.email_to:
        send_progress_email(
            args,
            f"[start] YOLO data generation {done}/{args.num_images}",
            build_start_body(args, done, chunks),
        )

    stop_event = threading.Event()
    with tqdm(total=args.num_images, initial=done, unit="img", dynamic_ncols=True, desc="YOLO generation", ascii=True) as progress:
        monitor = threading.Thread(
            target=monitor_progress,
            args=(args, progress, stop_event, start, initial_done),
            daemon=True,
        )
        monitor.start()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                pending_jobs = deque((worker_idx, ids, 0) for worker_idx, ids in enumerate(chunks))
                running = {}

                def submit_available():
                    while pending_jobs and len(running) < args.workers:
                        job_worker_idx, job_ids, attempt = pending_jobs.popleft()
                        future = executor.submit(run_chunk, args, job_worker_idx, job_ids, attempt)
                        running[future] = (job_worker_idx, job_ids, attempt)

                submit_available()
                while running:
                    completed_futures, _ = concurrent.futures.wait(
                        running,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in completed_futures:
                        worker_idx, ids, attempt = running.pop(future)
                        try:
                            worker_idx, size = future.result()
                            tqdm.write(f"worker {worker_idx} finished {size} ids")
                        except Exception as exc:
                            remaining_ids = unfinished_ids(args, ids)
                            done_count = len(ids) - len(remaining_ids)
                            if remaining_ids and attempt < args.max_worker_retries:
                                next_attempt = attempt + 1
                                tqdm.write(
                                    f"worker {worker_idx} failed after {done_count}/{len(ids)} ids: {exc}; "
                                    f"retry {next_attempt}/{args.max_worker_retries} for {len(remaining_ids)} unfinished ids"
                                )
                                if args.retry_delay_seconds > 0:
                                    time.sleep(args.retry_delay_seconds)
                                pending_jobs.append((worker_idx, remaining_ids, next_attempt))
                            elif remaining_ids:
                                raise RuntimeError(
                                    f"worker {worker_idx} exhausted retries; "
                                    f"{len(remaining_ids)}/{len(ids)} ids unfinished"
                                ) from exc
                            else:
                                tqdm.write(f"worker {worker_idx} failed after finishing its ids: {exc}")

                        submit_available()

                    done = min(existing_count(args.output), args.num_images)
                    if done > progress.n:
                        progress.update(done - progress.n)
                    elapsed = time.time() - start
                    rate, generated, _ = run_rate(done, initial_done, start)
                    eta = (args.num_images - done) / rate / 60 if rate > 0 else 0
                    progress.set_postfix({
                        "img/s": f"{rate:.2f}",
                        "run": generated,
                        "ETA": f"{eta:.1f}m" if rate > 0 else "calc",
                    }, refresh=True)
            done = min(existing_count(args.output), args.num_images)
            if args.email_to:
                send_progress_email(
                    args,
                    f"[완료] YOLO 데이터 생성 {done}/{args.num_images}",
                    build_progress_body(args, done, start, initial_done),
                )
        except Exception as exc:
            done = min(existing_count(args.output), args.num_images)
            if args.email_to:
                send_progress_email(
                    args,
                    f"[오류] YOLO 데이터 생성 중단 {done}/{args.num_images}",
                    build_progress_body(args, done, start, initial_done) + f"\n오류: {exc}\n",
                )
            raise
        finally:
            stop_event.set()
            monitor.join(timeout=2.0)
            done = min(existing_count(args.output), args.num_images)
            if done > progress.n:
                progress.update(done - progress.n)


if __name__ == "__main__":
    main()
