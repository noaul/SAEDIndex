"""Read-only, recomputed evidence for grouping and fixed-scale CIF comparisons."""
import numpy as np

from .pattern import lattice_occupancy

SCORE_LABELS = {"ratio": "长度比", "scale": "共同尺度", "angle": "夹角",
                "forbidden": "消光冲突", "dynamic": "动力学可能",
                "missing": "未见预测峰", "zone": "晶带轴先验"}


def group_diagnostics(session, group):
    lattice = group.lattice
    members = set(lattice.members)
    previous = set()
    for other in session.sets:
        if other is group:
            break
        previous.update(other.lattice.members)
    new = members - previous
    rows, errors = [], []
    found = group.indices() if group.candidate else None
    for i, spot in enumerate(session.spots):
        node = lattice.members.get(i)
        row = {"spot": i, "member": node is not None, "residual_px": None, "d_error": None,
               "status": "本组外"}
        if node is not None:
            residual = float(np.linalg.norm(spot.xy - lattice.position(*node)))
            radius = float(np.linalg.norm(spot.xy - lattice.center))
            tolerance = max(2.5, .02 * radius)
            row.update(residual_px=residual,
                       status="偏差较大" if residual > .8 * tolerance else "支持")
            if i in previous:
                row["status"] += " / 共享"
            if found is not None and session.px_per_unit and radius > 0:
                hkl = node[0] * found[0] + node[1] * found[1]
                if np.any(hkl):
                    row["d_error"] = float(session.px_per_unit / radius /
                                           group.candidate.phase.d_spacing(hkl) - 1)
            errors.append(residual)
        rows.append(row)
    directions = np.array([session.spots[i].xy - lattice.center for i in new])
    spread = 0.
    if len(directions) >= 2:
        singular = np.linalg.svd(directions, compute_uv=False)
        spread = float(singular[-1] / max(singular[0], 1e-12))
    closure = None
    if all(session.has_spot(group, n) for n in ((1, 0), (0, 1), (1, 1))):
        a, b, ab = (session.node_xy(group, n) - lattice.center
                    for n in ((1, 0), (0, 1), (1, 1)))
        closure = float(np.linalg.norm(ab - a - b))
    warnings = []
    if previous and len(new) < 6:
        warnings.append("与已有组重复或独立新增斑点不足 6 个，请核对是否需要此组。")
    if len(members) < 6 or spread < .08:
        warnings.append("独立二维晶格证据不足，请核对中心和斑点分布。")
    c = group.candidate
    suggestion, gap = None, None
    if c is not None and session.px_per_unit:
        suggestion = float(session.px_per_unit / c.scale) if c.scale > 0 else None
        alternatives = [other.cost for other in group.candidates if other is not c]
        gap = min(alternatives) - c.cost if alternatives else None
        if gap is not None and abs(gap) < .4:
            warnings.append("候选评分接近，物相或晶带轴仍有歧义。")
        elif gap is not None and gap < 0:
            warnings.append("当前选中候选不是最低评分结果，请核对选择依据。")
        if abs(c.ratio_err) > session.length_tol or abs(c.angle_meas - c.angle_theo) > session.angle_tol:
            warnings.append("角度或长度比超出容差，调整共同尺度不能消除该几何冲突。")
        if abs(c.scale_err) > session.scale_tol:
            warnings.append("共同尺度偏差超出容差，请核对原标尺。")
        geometry = sum(c.cost_parts.get(k, 0.) for k in ("ratio", "scale", "angle"))
        intensity = c.cost_parts.get("missing", 0.)
        if intensity > geometry and intensity > .5:
            warnings.append("未见预测峰项影响较大；近似强度不能单独确认物相。")
    if session.image and session.image.calibration.conflict:
        warnings.append("DM 与图中标尺冲突，需人工定标。")
    if session.image and session.image.calibration.source.startswith("CIF 辅助"):
        warnings.append("比例来自当前 CIF，不构成独立物相验证。")
    usable = sum(np.linalg.norm(p.xy - lattice.center) > 3 for p in session.spots)
    return {"support": len(members), "new_support": len(new),
            "shared_support": len(members & previous), "coverage": len(members) / max(usable, 1),
            "occupancy": min(1., lattice_occupancy(lattice, session.spots)) if members else 0.,
            "rms_px": float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.,
            "max_residual_px": max(errors, default=0.), "spread": spread,
            "closure_px": closure, "scale_suggestion": suggestion,
            "candidate_gap": gap, "warnings": warnings, "spots": rows,
            "cost_parts": dict(c.cost_parts) if c is not None else {}}


def simulation_matches(session, group, reflections):
    """Match by assigned indices, never nearest unlabelled spot or fitted scale."""
    found = group.indices()
    observed = {}
    if found is not None:
        for i, node in group.lattice.members.items():
            hkl = tuple(node[0] * found[0] + node[1] * found[1])
            observed.setdefault(hkl, []).append(i)
    result = []
    for reflection in reflections:
        hkl, x, y, intensity, allowed = reflection
        ids = observed.get(tuple(hkl), [])
        xy = np.array([x, y])
        spot_id = min(ids, key=lambda i: np.linalg.norm(session.spots[i].xy - xy)) if ids else None
        error = float(np.linalg.norm(session.spots[spot_id].xy - xy)) if spot_id is not None else None
        tolerance = max(2.5, .02 * float(np.linalg.norm(xy - group.lattice.center)))
        result.append({"reflection": reflection, "spot": spot_id, "residual_px": error,
                       "matched": error is not None and error <= tolerance})
    return result
