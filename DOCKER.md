# pyiwfm Docker Setup

This directory contains Docker configuration for running pyiwfm with 3D web visualization.

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and start the web viewer
docker-compose up --build

# Access the viewer at http://localhost:8080
```

### Using Docker directly

```bash
# Build the image
docker build -t pyiwfm .

# Run with your model mounted
docker run -p 8080:8080 -v /path/to/your/model:/model pyiwfm
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8080 | Web server port |
| `TITLE` | auto-detect | Viewer title (auto-detected from model name if not set) |
| `THEME` | "light" | UI theme ("light" or "dark") |
| `MODE` | "web" | "web" for viewer, "export" for VTK/GeoPackage export |
| `MODEL_PATH` | "/model" | Path to model directory inside the container |

### Volumes

| Container Path | Description |
|----------------|-------------|
| `/model` | Mount your IWFM model directory here (read-only) |
| `/output` | Output directory for exported files |

## Usage Examples

### View the sample model

```bash
# Using docker-compose (model path already configured)
docker-compose up --build
```

### View your own model

```bash
docker run -p 8080:8080 \
  -v /path/to/your/model:/model:ro \
  -e TITLE="My Model" \
  pyiwfm
```

### Export model to VTK/GeoPackage

```bash
docker run \
  -v /path/to/your/model:/model:ro \
  -v /path/to/output:/output \
  -e MODE=export \
  pyiwfm
```

### Run with dark theme

```bash
docker run -p 8080:8080 \
  -v /path/to/model:/model:ro \
  -e THEME=dark \
  pyiwfm
```

### Run on different port

```bash
docker run -p 9000:9000 \
  -v /path/to/model:/model:ro \
  -e PORT=9000 \
  pyiwfm
```

## Expected Model Directory Structure

The container expects an IWFM model directory with the standard structure:

```
model/
├── Preprocessor/
│   ├── PreProcessor_MAIN.IN   # Required - main input file
│   ├── NodeXY.dat             # Node coordinates
│   ├── Element.dat            # Element definitions
│   ├── Strata.dat             # Stratigraphy
│   ├── Stream.dat             # Stream network (optional)
│   └── Lake.dat               # Lakes (optional)
├── Simulation/                 # Optional
│   └── ...
└── Results/                    # Optional
    └── ...
```

## Development Mode

For development with Jupyter notebooks:

```bash
docker-compose --profile dev up dev

# Access Jupyter at http://localhost:8888
# Web viewer at http://localhost:8081
```

## Troubleshooting

### "Could not find preprocessor input file"

Make sure your model directory contains `Preprocessor/PreProcessor_MAIN.IN` or similar.

### Viewer shows blank screen

This can happen if the browser doesn't support WebGL. Try:
- Using Chrome or Firefox
- Checking browser WebGL support at https://get.webgl.org/

### Container exits immediately

Check the logs:
```bash
docker-compose logs
# or
docker logs <container_id>
```

### Performance issues with large models

For models with >50,000 nodes, increase memory:
```yaml
# In docker-compose.yaml
deploy:
  resources:
    limits:
      memory: 8G
```
