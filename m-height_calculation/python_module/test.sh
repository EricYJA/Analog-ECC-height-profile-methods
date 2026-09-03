#Install Release
python -m pip install . -v \
  --config-settings=cmake.build-type=Release \
  --config-settings=cmake.define.SANITIZE=OFF \
  --config-settings=cmake.define.CMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  --config-settings=cmake.define.pybind11_DIR="$(python -m pybind11 --cmakedir)"

# Install Debug (if needed)
python -m pip install . -v \
  --config-settings=cmake.build-type=Debug \
  --config-settings=cmake.define.SANITIZE=ON \
  --config-settings=cmake.define.CMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  --config-settings=cmake.define.pybind11_DIR="$(python -m pybind11 --cmakedir)"


python -m pip install . -v \
  --config-settings=cmake.build-type=Release \
  --config-settings=cmake.define.SANITIZE=OFF \
  --config-settings=cmake.define.CMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  --config-settings=cmake.define.pybind11_DIR="$(python -m pybind11 --cmakedir)"

# 1) Confirm which interpreter is running
which python
python -c "import sys; print(sys.executable); print(sys.version)"

# 2) Confirm the package is installed *for that interpreter*
python -m pip show -f solve_m_height_cpp

# 3) Show where Python looks for packages
python - <<'PY'
import site, sys, glob
print("exe:", sys.executable)
print("site:", site.getsitepackages())
print("usersite:", site.getusersitepackages())
candidates = []
for p in site.getsitepackages()+[site.getusersitepackages()]:
    candidates += glob.glob(p + "/solve_m_height_cpp*.so")
print("found:", candidates)
PY