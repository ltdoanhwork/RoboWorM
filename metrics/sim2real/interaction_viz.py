"""Visual evidence for the sim2real interaction metric.

The plan calls this the only proof that "the metric really catches it", so the
panels are built to be falsifiable by eye: every frame the metric flags is drawn
with the masks it used, the gap it measured, and the contact state it inferred,
side by side with GT.

Two products:

``episode_panel``
    A single PNG per episode: a strip of sampled frames (GT row / pred row) with
    mask overlays, plus a plot of gap(t) with the contact threshold, the contact
    timelines of both clips, and markers on the flagged frames.

``violation_clip``
    An mp4 cut around the worst flagged frame, so a violation can be watched
    rather than inferred from a number.
"""

import os

import cv2
import numpy as np

ROLE_BGR = {"object": (0, 220, 0), "gripper": (60, 60, 255), "target": (255, 160, 60)}
FLAG_BGR = (0, 0, 255)
OK_BGR = (0, 200, 0)


def overlay_masks(frame_bgr, masks_by_role, alpha=0.45):
    out = frame_bgr.copy()
    for role in ("target", "gripper", "object"):  # object drawn last = on top
        m = masks_by_role.get(role)
        if m is None or not np.any(m):
            continue
        if m.shape[:2] != out.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (out.shape[1], out.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        tint = np.zeros_like(out)
        tint[:] = ROLE_BGR[role]
        out = np.where(m[..., None], ((1 - alpha) * out + alpha * tint).astype(np.uint8), out)
    return out


def _draw_gap_line(img, mask_o, mask_g):
    """Draw the segment realising the object<->gripper gap, for visual auditing."""
    if not np.any(mask_o) or not np.any(mask_g):
        return img
    if mask_o.shape[:2] != img.shape[:2]:
        mask_o = cv2.resize(mask_o.astype(np.uint8), (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        mask_g = cv2.resize(mask_g.astype(np.uint8), (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    oy, ox = np.nonzero(mask_o)
    gy, gx = np.nonzero(mask_g)
    if oy.size == 0 or gy.size == 0:
        return img
    # subsample for speed; we only need a representative closest pair
    step_o = max(1, oy.size // 400)
    step_g = max(1, gy.size // 400)
    oy, ox = oy[::step_o], ox[::step_o]
    gy, gx = gy[::step_g], gx[::step_g]
    d = (oy[:, None] - gy[None, :]) ** 2 + (ox[:, None] - gx[None, :]) ** 2
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    cv2.line(img, (int(ox[i]), int(oy[i])), (int(gx[j]), int(gy[j])), (0, 255, 255), 1)
    return img


def _plot_curves(gap_pred, gap_gt, contact_pred, contact_gt, tau_gap,
                 flagged, width, height=220):
    """Matplotlib-free curve panel: gap(t) + contact timelines."""
    panel = np.full((height, width, 3), 26, np.uint8)
    T = max(len(gap_pred), 1)
    plot_h = height - 60
    finite = np.concatenate([np.asarray(gap_pred, float), np.asarray(gap_gt, float)])
    finite = finite[np.isfinite(finite)]
    ymax = float(np.percentile(finite, 98)) if finite.size else 1.0
    ymax = max(ymax, tau_gap * 2.0, 1e-3)

    def to_xy(t, v):
        x = int(round(t * (width - 1) / max(1, T - 1)))
        y = plot_h - int(round(min(v, ymax) / ymax * (plot_h - 12))) - 2
        return x, y

    # threshold line
    _, ty = to_xy(0, tau_gap)
    cv2.line(panel, (0, ty), (width - 1, ty), (90, 90, 200), 1, cv2.LINE_AA)
    cv2.putText(panel, f"tau_gap={tau_gap:g}", (4, max(12, ty - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 235), 1)

    for series, color, label in ((gap_gt, (160, 160, 160), "gap GT"),
                                 (gap_pred, (80, 220, 255), "gap pred")):
        pts = [to_xy(t, v) for t, v in enumerate(np.asarray(series, float))
               if np.isfinite(v)]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(panel, a, b, color, 1, cv2.LINE_AA)
        _ = label

    # contact timelines: bright = contact, dim blue = unknown (occluded), blank = none
    for row, (c, color, name) in enumerate((
            (contact_gt, (160, 160, 160), "contact GT"),
            (contact_pred, (80, 220, 255), "contact pred"))):
        y0 = plot_h + 6 + row * 16
        c = np.asarray(c, np.int8)
        for t in range(len(c)):
            if c[t] == 0:
                continue
            x0 = int(round(t * (width - 1) / max(1, T - 1)))
            x1 = int(round((t + 1) * (width - 1) / max(1, T - 1)))
            col = color if c[t] == 1 else (150, 90, 40)
            cv2.rectangle(panel, (x0, y0), (max(x1, x0 + 1), y0 + 10), col, -1)
        cv2.putText(panel, name, (4, y0 + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (230, 230, 230), 1)
    cv2.putText(panel, "dim blue = occluded (unknown)", (width - 220, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 90, 40), 1)

    # flagged frames as red ticks along the top
    for t in np.atleast_1d(np.asarray(flagged, int)):
        if 0 <= t < T:
            x = int(round(t * (width - 1) / max(1, T - 1)))
            cv2.line(panel, (x, 0), (x, 8), FLAG_BGR, 1)
    cv2.putText(panel, "red ticks = flagged frames", (width - 220, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, FLAG_BGR, 1)
    return panel


def episode_panel(view, metrics, out_path, num_samples=7, tile=(320, 192),
                  scale=2, prefer_flagged=True):
    """Render the GT/pred comparison panel for one view of one episode.

    ``view`` is one entry of the tracker's ``views`` list (needs ``frames_dir``
    and the per-role masks); ``metrics`` is the dict from ``episode_metrics``.
    """
    T = len(view["gt"]["object"])
    flagged = [t for t in metrics.get("levitation_frames", []) if 0 <= t < T]
    ts = list(np.linspace(0, T - 1, num_samples).round().astype(int))
    if prefer_flagged and flagged:
        # make sure the viewer actually sees the frames the metric complains about
        keep = ts[: max(1, num_samples - min(3, len(flagged)))]
        picks = list(np.asarray(flagged)[np.linspace(
            0, len(flagged) - 1, min(3, len(flagged))).round().astype(int)])
        ts = sorted(set(keep) | set(int(x) for x in picks))

    tw, th = tile[0] * scale, tile[1] * scale
    rows = []
    for tag in ("gt", "pred"):
        contact = metrics["contact_gt"] if tag == "gt" else metrics["contact_pred"]
        panels = []
        for t in ts:
            f = os.path.join(view["frames_dir"][tag], f"{t:05d}.jpg")
            img = cv2.imread(f)
            if img is None:
                img = np.zeros((th, tw, 3), np.uint8)
            img = overlay_masks(img, {r: view[tag][r][t] for r in
                                      ("object", "gripper", "target")})
            img = _draw_gap_line(img, view[tag]["object"][t], view[tag]["gripper"][t])
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            state = int(contact[t]) if t < len(contact) else 0
            on, unk = state == 1, state == -1
            border = FLAG_BGR if (tag == "pred" and t in flagged) else (
                OK_BGR if on else ((150, 90, 40) if unk else (70, 70, 70)))
            cv2.rectangle(img, (0, 0), (tw - 1, th - 1), border, 3)
            label = f"{tag} t{t}" + (" C" if on else (" ?" if unk else ""))
            if tag == "pred" and t in flagged:
                label += " LEVIT"
            cv2.putText(img, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2)
            panels.append(img)
        rows.append(np.hstack(panels))

    strip = np.vstack(rows)
    curves = _plot_curves(
        metrics.get("_gap_pred", []), metrics.get("_gap_gt", []),
        metrics["contact_pred"], metrics["contact_gt"],
        metrics["params"]["tau_gap"], flagged, strip.shape[1])

    header = np.full((58, strip.shape[1], 3), 18, np.uint8)
    sc = metrics["sim2real_interaction_score"]
    sc_txt = "INVALID" if sc is None else f"{sc:.3f}"
    vio = metrics["violations"]
    agr = metrics["agreement"]
    line1 = (f"score={sc_txt}  levit={vio.get('levitation_rate', float('nan')):.3f}"
             f" (gt {vio.get('levitation_rate_gt', float('nan')):.3f})"
             f"  grasp_ratio={vio.get('grasp_follow_ratio', float('nan')):.2f}"
             f"  present_deficit={vio.get('object_present_deficit', float('nan')):.3f}")
    onset = agr.get("contact_onset_err_frames")
    line2 = (f"onset_err={onset}f  tIoU={agr.get('contact_temporal_iou', float('nan')):.2f}"
             f"  traj_err={agr.get('obj_traj_err', float('nan')):.2f}"
             f"  gap_err={agr.get('gap_curve_err', float('nan')):.2f}"
             f"  steps={metrics['step_level']['num_passed']}/"
             f"{metrics['step_level']['num_nodes']}"
             f"  view={view['view_name']}")
    cv2.putText(header, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1)
    cv2.putText(header, line2, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1)

    out = np.vstack([header, strip, curves])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, out)
    return out_path


def _hud_strip(metrics, t, width, height=72):
    """Live read-out under one episode row: gap curve with a playhead + state.

    A still panel proves the metric looked at the right frames; a moving playhead on
    the gap curve proves it reacted at the right *time*, which is the claim the
    contact-timing and levitation terms actually make.
    """
    panel = np.full((height, width, 3), 22, np.uint8)
    gp = np.asarray(metrics.get("_gap_pred", []), float)
    gg = np.asarray(metrics.get("_gap_gt", []), float)
    tau = metrics["params"]["tau_gap"]
    T = max(len(gp), 1)
    curve_w = int(width * 0.62)
    plot_h = height - 22

    fin = np.concatenate([gp, gg])
    fin = fin[np.isfinite(fin)]
    ymax = max(float(np.percentile(fin, 98)) if fin.size else 1.0, tau * 2.0, 1e-3)

    def xy(i, v):
        x = int(round(i * (curve_w - 1) / max(1, T - 1)))
        y = plot_h - int(round(min(v, ymax) / ymax * (plot_h - 8))) - 2
        return x, y

    _, ty = xy(0, tau)
    cv2.line(panel, (0, ty), (curve_w - 1, ty), (90, 90, 200), 1)
    for series, color in ((gg, (150, 150, 150)), (gp, (80, 220, 255))):
        pts = [xy(i, v) for i, v in enumerate(series) if np.isfinite(v)]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(panel, a, b, color, 1, cv2.LINE_AA)

    flagged = set(metrics.get("levitation_frames", []))
    for f in flagged:
        if 0 <= f < T:
            x, _ = xy(f, 0)
            cv2.line(panel, (x, 0), (x, 6), FLAG_BGR, 1)

    px, _ = xy(min(t, T - 1), 0)
    cv2.line(panel, (px, 0), (px, plot_h), (255, 255, 255), 1)
    cv2.putText(panel, "gap(t): white=now, cyan=pred, grey=GT, red tick=flag",
                (2, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 170), 1)

    cp = np.asarray(metrics["contact_pred"], np.int8)
    cg = np.asarray(metrics["contact_gt"], np.int8)
    sp = int(cp[t]) if t < len(cp) else 0
    sg = int(cg[t]) if t < len(cg) else 0
    txt = {1: "CONTACT", 0: "no contact", -1: "occluded"}
    col = {1: OK_BGR, 0: (150, 150, 150), -1: (150, 90, 40)}
    x0 = curve_w + 10
    cv2.putText(panel, f"GT  : {txt[sg]}", (x0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, col[sg], 1)
    cv2.putText(panel, f"pred: {txt[sp]}", (x0, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, col[sp], 1)
    gv = gp[t] if t < len(gp) and np.isfinite(gp[t]) else None
    gtv = gg[t] if t < len(gg) and np.isfinite(gg[t]) else None
    cv2.putText(panel,
                f"gap pred={'--' if gv is None else f'{gv:.2f}'}  "
                f"GT={'--' if gtv is None else f'{gtv:.2f}'} (diam)",
                (x0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 210, 210), 1)
    if t in flagged:
        cv2.putText(panel, "LEVITATION", (width - 130, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, FLAG_BGR, 2)
    return panel


def _episode_row(view, metrics, t, label, tile, scale, banner_col):
    """One episode as [GT | pred] with overlays, a title bar and the live HUD."""
    tw, th = tile[0] * scale, tile[1] * scale
    cells = []
    for tag in ("gt", "pred"):
        f = os.path.join(view["frames_dir"][tag], f"{t:05d}.jpg")
        img = cv2.imread(f)
        if img is None:
            img = np.zeros((th, tw, 3), np.uint8)
        n = len(view[tag]["object"])
        ti = min(t, n - 1)
        img = overlay_masks(img, {r: view[tag][r][ti] for r in
                                  ("object", "gripper", "target")})
        img = _draw_gap_line(img, view[tag]["object"][ti], view[tag]["gripper"][ti])
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        contact = metrics["contact_gt"] if tag == "gt" else metrics["contact_pred"]
        state = int(contact[t]) if t < len(contact) else 0
        flag = (tag == "pred" and t in set(metrics.get("levitation_frames", [])))
        border = FLAG_BGR if flag else (OK_BGR if state == 1 else
                                        ((150, 90, 40) if state == -1 else (70, 70, 70)))
        cv2.rectangle(img, (0, 0), (tw - 1, th - 1), border, 3)
        cv2.putText(img, f"{tag.upper()}  t={t}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cells.append(img)
    body = np.hstack(cells)

    bar = np.full((30, body.shape[1], 3), 16, np.uint8)
    cv2.rectangle(bar, (0, 0), (6, 29), banner_col, -1)
    cv2.putText(bar, label, (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, banner_col, 1)
    return np.vstack([bar, body, _hud_strip(metrics, t, body.shape[1])])


def comparison_clip(entries, out_path, fps=5, tile=(320, 192), scale=2):
    """Stack several episodes into ONE video: violating on top, clean below.

    ``entries`` is a list of ``{"view", "metrics", "label", "color"}``. Rows are
    played in lockstep and short clips hold their last frame, so the eye compares
    the same moment across episodes. Seeing a low-scoring and a high-scoring episode
    side by side is what makes the score meaningful - a violation number alone gives
    no sense of what "clean" looks like on the same tracker.
    """
    T = max(len(e["metrics"]["contact_pred"]) for e in entries)
    rows0 = [_episode_row(e["view"], e["metrics"], 0, e["label"], tile, scale,
                          e.get("color", (255, 255, 255))) for e in entries]
    W = max(r.shape[1] for r in rows0)
    H = sum(r.shape[0] + 8 for r in rows0)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for t in range(T):
        stack = []
        for e in entries:
            n = len(e["metrics"]["contact_pred"])
            row = _episode_row(e["view"], e["metrics"], min(t, n - 1), e["label"],
                               tile, scale, e.get("color", (255, 255, 255)))
            if row.shape[1] != W:
                pad = np.full((row.shape[0], W - row.shape[1], 3), 16, np.uint8)
                row = np.hstack([row, pad])
            stack.append(row)
            stack.append(np.full((8, W, 3), 8, np.uint8))
        vw.write(np.vstack(stack)[:H])
    vw.release()
    return out_path


def violation_clip(view, metrics, out_path, fps=5, pad=6, tile=(320, 192), scale=2):
    """Write an mp4 around the flagged frames: GT on the left, pred on the right."""
    T = len(view["gt"]["object"])
    flagged = [t for t in metrics.get("levitation_frames", []) if 0 <= t < T]
    if flagged:
        lo, hi = max(0, min(flagged) - pad), min(T - 1, max(flagged) + pad)
    else:
        lo, hi = 0, T - 1

    tw, th = tile[0] * scale, tile[1] * scale
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (tw * 2, th))
    for t in range(lo, hi + 1):
        cells = []
        for tag in ("gt", "pred"):
            f = os.path.join(view["frames_dir"][tag], f"{t:05d}.jpg")
            img = cv2.imread(f)
            if img is None:
                img = np.zeros((th, tw, 3), np.uint8)
            img = overlay_masks(img, {r: view[tag][r][t] for r in
                                      ("object", "gripper", "target")})
            img = _draw_gap_line(img, view[tag]["object"][t], view[tag]["gripper"][t])
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            contact = metrics["contact_gt"] if tag == "gt" else metrics["contact_pred"]
            state = int(contact[t]) if t < len(contact) else 0
            on, unk = state == 1, state == -1
            flag = (tag == "pred" and t in flagged)
            cv2.rectangle(img, (0, 0), (tw - 1, th - 1),
                          FLAG_BGR if flag else (OK_BGR if on else
                                                ((150, 90, 40) if unk else (70, 70, 70))), 3)
            cv2.putText(img, f"{tag} t{t}" + (" CONTACT" if on else (" OCCLUDED" if unk else ""))
                        + (" LEVITATION" if flag else ""),
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cells.append(img)
        vw.write(np.hstack(cells))
    vw.release()
    return out_path
