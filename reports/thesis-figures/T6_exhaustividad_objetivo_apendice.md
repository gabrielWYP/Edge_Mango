Tabla T6. Puntos de operación orientados a exhaustividad, con umbral elegido solo en validación de cada lote (el mayor umbral con exhaustividad por foto en val >= 0.90 o 0.95; o F2 máximo) y aplicado al prueba de ese lote. Si ningún umbral alcanza el objetivo se usa el menor de la rejilla y se marca (fila «objetivo alcanzado»). Reglas definidas a posteriori; descriptivas, fuera de H1/H2. Apéndice: todos los brazos, incluidos los exploratorios.

| Brazo | Métrica | n folds | Media ± sd | Lote 1 | Lote 2 | Lote 3 | Lote 4 | Lote 5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Maestro RGB | recall >= 0.90 en val: umbral elegido en val | 5 | 0.40 ± 0.04 | 0.44 | 0.42 | 0.35 | 0.42 | 0.38 |
| Maestro NIR | recall >= 0.90 en val: umbral elegido en val | 5 | 0.39 ± 0.04 | 0.43 | 0.40 | 0.36 | 0.42 | 0.35 |
| Maestro RGB+NIR | recall >= 0.90 en val: umbral elegido en val | 5 | 0.42 ± 0.02 | 0.44 | 0.40 | 0.41 | 0.42 | 0.45 |
| Maestro RGB+NIR, dos troncos (exploratorio) | recall >= 0.90 en val: umbral elegido en val | 5 | 0.41 ± 0.05 | 0.48 | 0.40 | 0.35 | 0.41 | 0.39 |
| Maestro RGB, solo cajas | recall >= 0.90 en val: umbral elegido en val | 5 | 0.39 ± 0.02 | 0.41 | 0.37 | 0.37 | 0.40 | 0.39 |
| Alumno RGB, CrossKD desde RGB | recall >= 0.90 en val: umbral elegido en val | 5 | 0.42 ± 0.02 | 0.45 | 0.41 | 0.39 | 0.42 | 0.41 |
| Alumno RGB, CrossKD desde RGB+NIR | recall >= 0.90 en val: umbral elegido en val | 5 | 0.40 ± 0.03 | 0.44 | 0.39 | 0.36 | 0.42 | 0.41 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | recall >= 0.90 en val: umbral elegido en val | 5 | 0.41 ± 0.03 | 0.43 | 0.44 | 0.37 | 0.41 | 0.40 |
| Maestro RGB | recall >= 0.90 en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Maestro NIR | recall >= 0.90 en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Maestro RGB+NIR | recall >= 0.90 en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Maestro RGB+NIR, dos troncos (exploratorio) | recall >= 0.90 en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Maestro RGB, solo cajas | recall >= 0.90 en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Alumno RGB, CrossKD desde RGB | recall >= 0.90 en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Alumno RGB, CrossKD desde RGB+NIR | recall >= 0.90 en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | recall >= 0.90 en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Maestro RGB | recall >= 0.90 en val: exhaustividad en prueba | 5 | 0.864 ± 0.133 | 0.667 | 0.934 | 0.965 | 0.785 | 0.967 |
| Maestro NIR | recall >= 0.90 en val: exhaustividad en prueba | 5 | 0.876 ± 0.101 | 0.782 | 0.954 | 0.952 | 0.750 | 0.940 |
| Maestro RGB+NIR | recall >= 0.90 en val: exhaustividad en prueba | 5 | 0.884 ± 0.084 | 0.860 | 0.985 | 0.904 | 0.757 | 0.912 |
| Maestro RGB+NIR, dos troncos (exploratorio) | recall >= 0.90 en val: exhaustividad en prueba | 5 | 0.859 ± 0.060 | 0.767 | 0.871 | 0.882 | 0.848 | 0.929 |
| Maestro RGB, solo cajas | recall >= 0.90 en val: exhaustividad en prueba | 5 | 0.908 ± 0.065 | 0.835 | 0.982 | 0.955 | 0.848 | 0.919 |
| Alumno RGB, CrossKD desde RGB | recall >= 0.90 en val: exhaustividad en prueba | 5 | 0.862 ± 0.080 | 0.787 | 0.909 | 0.937 | 0.765 | 0.912 |
| Alumno RGB, CrossKD desde RGB+NIR | recall >= 0.90 en val: exhaustividad en prueba | 5 | 0.879 ± 0.071 | 0.857 | 0.949 | 0.952 | 0.787 | 0.849 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | recall >= 0.90 en val: exhaustividad en prueba | 5 | 0.890 ± 0.091 | 0.822 | 0.970 | 0.967 | 0.767 | 0.924 |
| Maestro RGB | recall >= 0.90 en val: especificidad en prueba | 5 | 0.210 ± 0.161 | 0.460 | 0.150 | 0.050 | 0.270 | 0.120 |
| Maestro NIR | recall >= 0.90 en val: especificidad en prueba | 5 | 0.236 ± 0.167 | 0.480 | 0.120 | 0.110 | 0.340 | 0.130 |
| Maestro RGB+NIR | recall >= 0.90 en val: especificidad en prueba | 5 | 0.218 ± 0.129 | 0.330 | 0.050 | 0.140 | 0.360 | 0.210 |
| Maestro RGB+NIR, dos troncos (exploratorio) | recall >= 0.90 en val: especificidad en prueba | 5 | 0.250 ± 0.117 | 0.450 | 0.260 | 0.170 | 0.190 | 0.180 |
| Maestro RGB, solo cajas | recall >= 0.90 en val: especificidad en prueba | 5 | 0.162 ± 0.100 | 0.220 | 0.040 | 0.080 | 0.280 | 0.190 |
| Alumno RGB, CrossKD desde RGB | recall >= 0.90 en val: especificidad en prueba | 5 | 0.250 ± 0.146 | 0.470 | 0.180 | 0.100 | 0.320 | 0.180 |
| Alumno RGB, CrossKD desde RGB+NIR | recall >= 0.90 en val: especificidad en prueba | 5 | 0.228 ± 0.117 | 0.320 | 0.100 | 0.110 | 0.350 | 0.260 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | recall >= 0.90 en val: especificidad en prueba | 5 | 0.166 ± 0.141 | 0.320 | 0.080 | 0.010 | 0.310 | 0.110 |
| Maestro RGB | recall >= 0.90 en val: tasa de descarte de sanas en prueba | 5 | 0.790 ± 0.161 | 0.540 | 0.850 | 0.950 | 0.730 | 0.880 |
| Maestro NIR | recall >= 0.90 en val: tasa de descarte de sanas en prueba | 5 | 0.764 ± 0.167 | 0.520 | 0.880 | 0.890 | 0.660 | 0.870 |
| Maestro RGB+NIR | recall >= 0.90 en val: tasa de descarte de sanas en prueba | 5 | 0.782 ± 0.129 | 0.670 | 0.950 | 0.860 | 0.640 | 0.790 |
| Maestro RGB+NIR, dos troncos (exploratorio) | recall >= 0.90 en val: tasa de descarte de sanas en prueba | 5 | 0.750 ± 0.117 | 0.550 | 0.740 | 0.830 | 0.810 | 0.820 |
| Maestro RGB, solo cajas | recall >= 0.90 en val: tasa de descarte de sanas en prueba | 5 | 0.838 ± 0.100 | 0.780 | 0.960 | 0.920 | 0.720 | 0.810 |
| Alumno RGB, CrossKD desde RGB | recall >= 0.90 en val: tasa de descarte de sanas en prueba | 5 | 0.750 ± 0.146 | 0.530 | 0.820 | 0.900 | 0.680 | 0.820 |
| Alumno RGB, CrossKD desde RGB+NIR | recall >= 0.90 en val: tasa de descarte de sanas en prueba | 5 | 0.772 ± 0.117 | 0.680 | 0.900 | 0.890 | 0.650 | 0.740 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | recall >= 0.90 en val: tasa de descarte de sanas en prueba | 5 | 0.834 ± 0.141 | 0.680 | 0.920 | 0.990 | 0.690 | 0.890 |
| Maestro RGB | recall >= 0.90 en val: precisión en prueba | 5 | 0.814 ± 0.011 | 0.831 | 0.812 | 0.801 | 0.811 | 0.814 |
| Maestro NIR | recall >= 0.90 en val: precisión en prueba | 5 | 0.821 ± 0.020 | 0.857 | 0.810 | 0.809 | 0.820 | 0.811 |
| Maestro RGB+NIR | recall >= 0.90 en val: precisión en prueba | 5 | 0.819 ± 0.014 | 0.837 | 0.803 | 0.807 | 0.826 | 0.821 |
| Maestro RGB+NIR, dos troncos (exploratorio) | recall >= 0.90 en val: precisión en prueba | 5 | 0.821 ± 0.016 | 0.848 | 0.823 | 0.808 | 0.807 | 0.818 |
| Maestro RGB, solo cajas | recall >= 0.90 en val: precisión en prueba | 5 | 0.812 ± 0.010 | 0.810 | 0.801 | 0.805 | 0.825 | 0.818 |
| Alumno RGB, CrossKD desde RGB | recall >= 0.90 en val: precisión en prueba | 5 | 0.822 ± 0.020 | 0.856 | 0.814 | 0.805 | 0.818 | 0.815 |
| Alumno RGB, CrossKD desde RGB+NIR | recall >= 0.90 en val: precisión en prueba | 5 | 0.820 ± 0.012 | 0.834 | 0.806 | 0.809 | 0.829 | 0.820 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | recall >= 0.90 en val: precisión en prueba | 5 | 0.810 ± 0.013 | 0.828 | 0.806 | 0.795 | 0.816 | 0.805 |
| Maestro RGB | recall >= 0.95 en val: umbral elegido en val | 5 | 0.38 ± 0.03 | 0.38 | 0.40 | 0.34 | 0.40 | 0.36 |
| Maestro NIR | recall >= 0.95 en val: umbral elegido en val | 5 | 0.36 ± 0.03 | 0.41 | 0.36 | 0.35 | 0.38 | 0.32 |
| Maestro RGB+NIR | recall >= 0.95 en val: umbral elegido en val | 5 | 0.40 ± 0.02 | 0.41 | 0.38 | 0.40 | 0.39 | 0.42 |
| Maestro RGB+NIR, dos troncos (exploratorio) | recall >= 0.95 en val: umbral elegido en val | 5 | 0.37 ± 0.05 | 0.45 | 0.37 | 0.32 | 0.38 | 0.35 |
| Maestro RGB, solo cajas | recall >= 0.95 en val: umbral elegido en val | 5 | 0.37 ± 0.02 | 0.39 | 0.35 | 0.35 | 0.38 | 0.37 |
| Alumno RGB, CrossKD desde RGB | recall >= 0.95 en val: umbral elegido en val | 5 | 0.39 ± 0.02 | 0.42 | 0.38 | 0.38 | 0.40 | 0.37 |
| Alumno RGB, CrossKD desde RGB+NIR | recall >= 0.95 en val: umbral elegido en val | 5 | 0.37 ± 0.02 | 0.39 | 0.36 | 0.34 | 0.38 | 0.37 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | recall >= 0.95 en val: umbral elegido en val | 5 | 0.38 ± 0.03 | 0.41 | 0.41 | 0.34 | 0.38 | 0.38 |
| Maestro RGB | recall >= 0.95 en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Maestro NIR | recall >= 0.95 en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Maestro RGB+NIR | recall >= 0.95 en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Maestro RGB+NIR, dos troncos (exploratorio) | recall >= 0.95 en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Maestro RGB, solo cajas | recall >= 0.95 en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Alumno RGB, CrossKD desde RGB | recall >= 0.95 en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Alumno RGB, CrossKD desde RGB+NIR | recall >= 0.95 en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | recall >= 0.95 en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Maestro RGB | recall >= 0.95 en val: exhaustividad en prueba | 5 | 0.936 ± 0.059 | 0.887 | 0.959 | 0.985 | 0.860 | 0.987 |
| Maestro NIR | recall >= 0.95 en val: exhaustividad en prueba | 5 | 0.932 ± 0.061 | 0.847 | 0.987 | 0.957 | 0.890 | 0.980 |
| Maestro RGB+NIR | recall >= 0.95 en val: exhaustividad en prueba | 5 | 0.935 ± 0.053 | 0.930 | 0.995 | 0.935 | 0.853 | 0.962 |
| Maestro RGB+NIR, dos troncos (exploratorio) | recall >= 0.95 en val: exhaustividad en prueba | 5 | 0.942 ± 0.036 | 0.880 | 0.949 | 0.965 | 0.945 | 0.970 |
| Maestro RGB, solo cajas | recall >= 0.95 en val: exhaustividad en prueba | 5 | 0.955 ± 0.038 | 0.897 | 0.995 | 0.982 | 0.953 | 0.947 |
| Alumno RGB, CrossKD desde RGB | recall >= 0.95 en val: exhaustividad en prueba | 5 | 0.920 ± 0.055 | 0.872 | 0.949 | 0.955 | 0.850 | 0.975 |
| Alumno RGB, CrossKD desde RGB+NIR | recall >= 0.95 en val: exhaustividad en prueba | 5 | 0.949 ± 0.029 | 0.930 | 0.980 | 0.982 | 0.920 | 0.935 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | recall >= 0.95 en val: exhaustividad en prueba | 5 | 0.947 ± 0.045 | 0.892 | 0.992 | 0.990 | 0.912 | 0.950 |
| Maestro RGB | recall >= 0.95 en val: especificidad en prueba | 5 | 0.100 ± 0.070 | 0.200 | 0.100 | 0.020 | 0.130 | 0.050 |
| Maestro NIR | recall >= 0.95 en val: especificidad en prueba | 5 | 0.144 ± 0.137 | 0.350 | 0.030 | 0.100 | 0.210 | 0.030 |
| Maestro RGB+NIR | recall >= 0.95 en val: especificidad en prueba | 5 | 0.118 ± 0.070 | 0.150 | 0.010 | 0.120 | 0.200 | 0.110 |
| Maestro RGB+NIR, dos troncos (exploratorio) | recall >= 0.95 en val: especificidad en prueba | 5 | 0.124 ± 0.116 | 0.320 | 0.140 | 0.050 | 0.070 | 0.040 |
| Maestro RGB, solo cajas | recall >= 0.95 en val: especificidad en prueba | 5 | 0.092 ± 0.059 | 0.140 | 0.020 | 0.040 | 0.150 | 0.110 |
| Alumno RGB, CrossKD desde RGB | recall >= 0.95 en val: especificidad en prueba | 5 | 0.144 ± 0.095 | 0.290 | 0.080 | 0.070 | 0.190 | 0.090 |
| Alumno RGB, CrossKD desde RGB+NIR | recall >= 0.95 en val: especificidad en prueba | 5 | 0.098 ± 0.050 | 0.110 | 0.050 | 0.050 | 0.170 | 0.110 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | recall >= 0.95 en val: especificidad en prueba | 5 | 0.098 ± 0.076 | 0.190 | 0.050 | 0.000 | 0.150 | 0.100 |
| Maestro RGB | recall >= 0.95 en val: tasa de descarte de sanas en prueba | 5 | 0.900 ± 0.070 | 0.800 | 0.900 | 0.980 | 0.870 | 0.950 |
| Maestro NIR | recall >= 0.95 en val: tasa de descarte de sanas en prueba | 5 | 0.856 ± 0.137 | 0.650 | 0.970 | 0.900 | 0.790 | 0.970 |
| Maestro RGB+NIR | recall >= 0.95 en val: tasa de descarte de sanas en prueba | 5 | 0.882 ± 0.070 | 0.850 | 0.990 | 0.880 | 0.800 | 0.890 |
| Maestro RGB+NIR, dos troncos (exploratorio) | recall >= 0.95 en val: tasa de descarte de sanas en prueba | 5 | 0.876 ± 0.116 | 0.680 | 0.860 | 0.950 | 0.930 | 0.960 |
| Maestro RGB, solo cajas | recall >= 0.95 en val: tasa de descarte de sanas en prueba | 5 | 0.908 ± 0.059 | 0.860 | 0.980 | 0.960 | 0.850 | 0.890 |
| Alumno RGB, CrossKD desde RGB | recall >= 0.95 en val: tasa de descarte de sanas en prueba | 5 | 0.856 ± 0.095 | 0.710 | 0.920 | 0.930 | 0.810 | 0.910 |
| Alumno RGB, CrossKD desde RGB+NIR | recall >= 0.95 en val: tasa de descarte de sanas en prueba | 5 | 0.902 ± 0.050 | 0.890 | 0.950 | 0.950 | 0.830 | 0.890 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | recall >= 0.95 en val: tasa de descarte de sanas en prueba | 5 | 0.902 ± 0.076 | 0.810 | 0.950 | 1.000 | 0.850 | 0.900 |
| Maestro RGB | recall >= 0.95 en val: precisión en prueba | 5 | 0.805 ± 0.007 | 0.816 | 0.808 | 0.800 | 0.798 | 0.805 |
| Maestro NIR | recall >= 0.95 en val: precisión en prueba | 5 | 0.813 ± 0.016 | 0.839 | 0.800 | 0.809 | 0.818 | 0.800 |
| Maestro RGB+NIR | recall >= 0.95 en val: precisión en prueba | 5 | 0.808 ± 0.006 | 0.814 | 0.798 | 0.808 | 0.810 | 0.811 |
| Maestro RGB+NIR, dos troncos (exploratorio) | recall >= 0.95 en val: precisión en prueba | 5 | 0.811 ± 0.016 | 0.838 | 0.813 | 0.801 | 0.803 | 0.800 |
| Maestro RGB, solo cajas | recall >= 0.95 en val: precisión en prueba | 5 | 0.807 ± 0.007 | 0.806 | 0.800 | 0.802 | 0.818 | 0.809 |
| Alumno RGB, CrossKD desde RGB | recall >= 0.95 en val: precisión en prueba | 5 | 0.811 ± 0.012 | 0.831 | 0.803 | 0.803 | 0.808 | 0.810 |
| Alumno RGB, CrossKD desde RGB+NIR | recall >= 0.95 en val: precisión en prueba | 5 | 0.807 ± 0.005 | 0.807 | 0.802 | 0.804 | 0.816 | 0.807 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | recall >= 0.95 en val: precisión en prueba | 5 | 0.807 ± 0.007 | 0.815 | 0.805 | 0.797 | 0.811 | 0.807 |
| Maestro RGB | F2 máximo en val: umbral elegido en val | 5 | 0.32 ± 0.01 | 0.33 | 0.31 | 0.31 | 0.32 | 0.33 |
| Maestro NIR | F2 máximo en val: umbral elegido en val | 5 | 0.29 ± 0.03 | 0.32 | 0.31 | 0.28 | 0.27 | 0.26 |
| Maestro RGB+NIR | F2 máximo en val: umbral elegido en val | 5 | 0.35 ± 0.03 | 0.30 | 0.35 | 0.38 | 0.33 | 0.37 |
| Maestro RGB+NIR, dos troncos (exploratorio) | F2 máximo en val: umbral elegido en val | 5 | 0.29 ± 0.05 | 0.38 | 0.29 | 0.25 | 0.30 | 0.25 |
| Maestro RGB, solo cajas | F2 máximo en val: umbral elegido en val | 5 | 0.31 ± 0.03 | 0.34 | 0.29 | 0.29 | 0.34 | 0.31 |
| Alumno RGB, CrossKD desde RGB | F2 máximo en val: umbral elegido en val | 5 | 0.30 ± 0.04 | 0.33 | 0.31 | 0.34 | 0.24 | 0.29 |
| Alumno RGB, CrossKD desde RGB+NIR | F2 máximo en val: umbral elegido en val | 5 | 0.29 ± 0.01 | 0.30 | 0.29 | 0.30 | 0.28 | 0.28 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | F2 máximo en val: umbral elegido en val | 5 | 0.31 ± 0.03 | 0.34 | 0.32 | 0.29 | 0.32 | 0.27 |
| Maestro RGB | F2 máximo en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Maestro NIR | F2 máximo en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Maestro RGB+NIR | F2 máximo en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Maestro RGB+NIR, dos troncos (exploratorio) | F2 máximo en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Maestro RGB, solo cajas | F2 máximo en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Alumno RGB, CrossKD desde RGB | F2 máximo en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Alumno RGB, CrossKD desde RGB+NIR | F2 máximo en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | F2 máximo en val: objetivo alcanzado en val (1 = sí) | 5 | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Maestro RGB | F2 máximo en val: exhaustividad en prueba | 5 | 0.993 ± 0.007 | 0.982 | 0.997 | 1.000 | 0.990 | 0.997 |
| Maestro NIR | F2 máximo en val: exhaustividad en prueba | 5 | 0.997 ± 0.003 | 0.995 | 1.000 | 1.000 | 0.998 | 0.995 |
| Maestro RGB+NIR | F2 máximo en val: exhaustividad en prueba | 5 | 0.993 ± 0.010 | 1.000 | 1.000 | 0.977 | 0.990 | 1.000 |
| Maestro RGB+NIR, dos troncos (exploratorio) | F2 máximo en val: exhaustividad en prueba | 5 | 0.998 ± 0.002 | 0.995 | 1.000 | 1.000 | 0.998 | 1.000 |
| Maestro RGB, solo cajas | F2 máximo en val: exhaustividad en prueba | 5 | 0.996 ± 0.008 | 0.982 | 1.000 | 1.000 | 0.998 | 1.000 |
| Alumno RGB, CrossKD desde RGB | F2 máximo en val: exhaustividad en prueba | 5 | 0.997 ± 0.004 | 0.995 | 0.992 | 1.000 | 1.000 | 1.000 |
| Alumno RGB, CrossKD desde RGB+NIR | F2 máximo en val: exhaustividad en prueba | 5 | 0.999 ± 0.001 | 0.997 | 0.997 | 1.000 | 1.000 | 1.000 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | F2 máximo en val: exhaustividad en prueba | 5 | 0.996 ± 0.005 | 0.997 | 1.000 | 0.997 | 0.988 | 1.000 |
| Maestro RGB | F2 máximo en val: especificidad en prueba | 5 | 0.012 ± 0.018 | 0.040 | 0.000 | 0.000 | 0.020 | 0.000 |
| Maestro NIR | F2 máximo en val: especificidad en prueba | 5 | 0.004 ± 0.009 | 0.000 | 0.000 | 0.020 | 0.000 | 0.000 |
| Maestro RGB+NIR | F2 máximo en val: especificidad en prueba | 5 | 0.020 ± 0.025 | 0.000 | 0.000 | 0.060 | 0.010 | 0.030 |
| Maestro RGB+NIR, dos troncos (exploratorio) | F2 máximo en val: especificidad en prueba | 5 | 0.002 ± 0.004 | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 |
| Maestro RGB, solo cajas | F2 máximo en val: especificidad en prueba | 5 | 0.004 ± 0.009 | 0.020 | 0.000 | 0.000 | 0.000 | 0.000 |
| Alumno RGB, CrossKD desde RGB | F2 máximo en val: especificidad en prueba | 5 | 0.006 ± 0.009 | 0.020 | 0.010 | 0.000 | 0.000 | 0.000 |
| Alumno RGB, CrossKD desde RGB+NIR | F2 máximo en val: especificidad en prueba | 5 | 0.006 ± 0.009 | 0.000 | 0.010 | 0.000 | 0.000 | 0.020 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | F2 máximo en val: especificidad en prueba | 5 | 0.002 ± 0.004 | 0.000 | 0.000 | 0.000 | 0.010 | 0.000 |
| Maestro RGB | F2 máximo en val: tasa de descarte de sanas en prueba | 5 | 0.988 ± 0.018 | 0.960 | 1.000 | 1.000 | 0.980 | 1.000 |
| Maestro NIR | F2 máximo en val: tasa de descarte de sanas en prueba | 5 | 0.996 ± 0.009 | 1.000 | 1.000 | 0.980 | 1.000 | 1.000 |
| Maestro RGB+NIR | F2 máximo en val: tasa de descarte de sanas en prueba | 5 | 0.980 ± 0.025 | 1.000 | 1.000 | 0.940 | 0.990 | 0.970 |
| Maestro RGB+NIR, dos troncos (exploratorio) | F2 máximo en val: tasa de descarte de sanas en prueba | 5 | 0.998 ± 0.004 | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 |
| Maestro RGB, solo cajas | F2 máximo en val: tasa de descarte de sanas en prueba | 5 | 0.996 ± 0.009 | 0.980 | 1.000 | 1.000 | 1.000 | 1.000 |
| Alumno RGB, CrossKD desde RGB | F2 máximo en val: tasa de descarte de sanas en prueba | 5 | 0.994 ± 0.009 | 0.980 | 0.990 | 1.000 | 1.000 | 1.000 |
| Alumno RGB, CrossKD desde RGB+NIR | F2 máximo en val: tasa de descarte de sanas en prueba | 5 | 0.994 ± 0.009 | 1.000 | 0.990 | 1.000 | 1.000 | 0.980 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | F2 máximo en val: tasa de descarte de sanas en prueba | 5 | 0.998 ± 0.004 | 1.000 | 1.000 | 1.000 | 0.990 | 1.000 |
| Maestro RGB | F2 máximo en val: precisión en prueba | 5 | 0.800 ± 0.003 | 0.803 | 0.797 | 0.799 | 0.802 | 0.798 |
| Maestro NIR | F2 máximo en val: precisión en prueba | 5 | 0.799 ± 0.002 | 0.799 | 0.798 | 0.802 | 0.800 | 0.798 |
| Maestro RGB+NIR | F2 máximo en val: precisión en prueba | 5 | 0.801 ± 0.003 | 0.800 | 0.798 | 0.805 | 0.800 | 0.804 |
| Maestro RGB+NIR, dos troncos (exploratorio) | F2 máximo en val: precisión en prueba | 5 | 0.799 ± 0.001 | 0.800 | 0.798 | 0.799 | 0.800 | 0.799 |
| Maestro RGB, solo cajas | F2 máximo en val: precisión en prueba | 5 | 0.799 ± 0.001 | 0.800 | 0.798 | 0.799 | 0.800 | 0.799 |
| Alumno RGB, CrossKD desde RGB | F2 máximo en val: precisión en prueba | 5 | 0.800 ± 0.002 | 0.802 | 0.798 | 0.799 | 0.800 | 0.799 |
| Alumno RGB, CrossKD desde RGB+NIR | F2 máximo en val: precisión en prueba | 5 | 0.800 ± 0.001 | 0.799 | 0.799 | 0.799 | 0.800 | 0.802 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | F2 máximo en val: precisión en prueba | 5 | 0.799 ± 0.001 | 0.799 | 0.798 | 0.798 | 0.800 | 0.799 |
