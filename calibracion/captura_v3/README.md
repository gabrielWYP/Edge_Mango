# captura_v3 ArUco pairs and RGB -> NIR homographies (lots 1-3)

One folder per capture day. Each folder has the ArUco calibration pair as `captura.py jornada` saved it (`calib-aruco-{rgb,nir,meta}-<ts>`), the homographies estimated from that pair, and the estimation report. The same files are in OCI: the pairs under `calib/` of each jornada, the matrices and reports under `homografias/<date>/` (lot 1: `homografias/20261024/lote1_oficial/`).

Board: ChArUco calib.io 8x11, 12 mm squares, 9 mm markers, DICT_4X4_50, placed at mango-top height. RGB 3264x2448 (Arducam), NIR 3280x2464 (NoIR).

**Convention** (same as `notebooks/matriz_homografia_aruco_3264x2448.npy`): `H = cv2.findHomography(pts_rgb, pts_nir)`, so the NIR is brought into the RGB frame with `cv2.warpPerspective(nir, np.linalg.inv(H), (3264, 2448))`.

## Which matrix to use

Use the `centros` matrix (marker centres, RANSAC 5 px). The `charuco` matrix (interpolated chessboard corners, RANSAC 3 px) has a slightly lower median error, but a heavier tail on lots 2 and 3; lot 1 and the scripts already use `centros`.

| Photos | Matrix | Median / p95 / max error (px) |
|---|---|---|
| Lot 1 `lote1-20261024`, all | `lote1-20261024/matriz_homografia_lote1_20261024_centros.npy` | 2.0 / 6.0 / 7.8 (n=37) |
| Lot 2 `lote2-20261025`, all | `lote2-20261025/matriz_homografia_aruco_3264x2448_lote2_20261025_centros.npy` | 2.7 / 6.2 / 20.7 (n=65) |
| Lot 3 `lote3-20261027`, up to ts 1793131668 (75-1 +6 h, 15:07:48) | `lote3-20261027/matriz_homografia_aruco_3264x2448_lote3_20261027_centros.npy` | 1.8 / 7.0 / 9.5 (n=70) |
| Lot 3, from ts 1793133155 (75-2 +6 h, 15:32:35) | `lote3-20261027/matriz_homografia_aruco_3264x2448_lote3_20261027_final_centros.npy` | 2.3 / 5.3 / 7.2 (n=70) |

Errors are on the ChArUco corners common to both images, in the RGB frame, on the board plane. Points below that plane (the mango flanks, the sheet) shift by parallax.

The matrices do not transfer between lots. The lot-1 matrix misses the lot-2 board by 32 px and the lot-3 morning board by 116 px (medians). The lot-3 morning matrix misses the lot-3 closing board by 53 px (median, max 118 px).

## Lot 3: two pairs

Around 15:25 the operator re-aimed the cameras, after a Pi reboot during the +6 h round. The morning pair (`calib-aruco-*-1793100324`, 06:25) is valid up to 75-1 +6 h. The closing pair (`calib-aruco-final-*-1793154347`, 21:25) was taken after the +12 h round. No photo was registered between 15:07:48 and 15:32:35.

A check on real photos agrees. The correlation between RGB gradients and warped NIR gradients over the mango area was:

- before the adjustment: 0.18-0.25 with the morning matrix, 0.02-0.07 with the closing one;
- in +12 h: 0.35-0.48 with the closing matrix, 0.06-0.08 with the morning one.

**Caveat.** With the closing matrix the fit is weaker on the first photos after the adjustment. The score was 0.0 to 0.02 at 75-2/3/4 +6 h, 0.10-0.22 at +8 h, 0.16-0.45 at +10 h and 0.35-0.48 at +12 h. Phase correlation puts the residual vertical offset at about -26 px at 75-3/4 +6 h, +10 to +20 px at +8 h and +3 px at +12 h. The mount seems to have settled over the afternoon. For pixel-level NIR/RGB work on lot 3 after 15:25, refine a per-photo translation on top of the closing matrix.

The `calib-aruco-final` type is not one of the `captura.py` calibration types. The pair was taken with the same session code (lights, camera controls, QC, metadata) from a helper script, which registers it in `jornada.json` under `calibracion["aruco-final"]`.

## Reports

`informe_homografia*.json` were written by `homografia_nueva.py` (lot-1 QA scripts, `data/cache/incoming/20261024/pruebas/scripts/`). In each report, the `vieja_22oct` entry is the comparison matrix passed in, not always the 22 Oct one:

- lot 1: the 22 Oct matrix;
- lot 2: the lot-1 matrix;
- lot 3 morning (`informe_homografia.json`): the lot-1 matrix (116 px median);
- lot 3 closing (`informe_homografia_final.json`): the lot-3 morning matrix.
