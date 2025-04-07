# Einops Implementation Sarvam Fellowship

This project implements a subset of the einops library functionality from scratch, focusing on the core `rearrange` operation for tensor manipulation. The implementation works with NumPy arrays and provides similar semantics to the original einops library.

## Features

The implementation supports:
- Reshaping operations
- Transposition
- Splitting of axes
- Merging of axes
- Ellipsis handling for batch dimensions

## Design Approach

The implementation follows a modular design:

1. **Pattern Parsing**: The `EinopsPatternParser` parses the einops pattern string into input and output specifications.

2. **Axis Specifications**: The `AxisSpec` class represents dimensions, handling both simple and composite axes.

3. **Validation and Normalization**: The `validate_and_normalize_pattern` function checks the pattern against the tensor shape and resolves dimension sizes.

4. **Tensor Transformation**: The `transform_tensor` function performs the actual tensor operations based on the parsed pattern.

## Proposed Approach
![image](https://github.com/user-attachments/assets/cdc87c86-52e1-4efc-bed0-e6fd9226353b)


## Usage Examples

```python
import numpy as np
from einops_implementation import rearrange

# Transpose
x = np.random.rand(3, 4)
result = rearrange(x, 'h w -> w h')

# Split an axis
x = np.random.rand(12, 10)
result = rearrange(x, '(h w) c -> h w c', h=3)

# Merge axes
x = np.random.rand(3, 4, 5)
result = rearrange(x, 'a b c -> (a b) c')

# Handle batch dimensions
x = np.random.rand(2, 3, 4, 5)
result = rearrange(x, '... h w -> ... (h w)')
```

## Limitations

The current implementation has some limitations compared to the full einops library:

1. **Repeating Axes**: The current implementation doesn't support repeating axes (e.g., `'a 1 c -> a b c'`). This would require additional functionality.

2. **Non-adjacent Merging**: The implementation requires axes to be adjacent when merging them in the output.

3. **Framework Support**: The implementation only supports NumPy arrays, not PyTorch, TensorFlow, or other frameworks.

## Running Tests

The implementation includes comprehensive unit tests with good coverage to ensure correctness:

```python
from einops_tests import run_tests

run_tests()
```

## Performance Considerations

The implementation aims for reasonable performance by:
1. Minimizing the number of intermediate tensor operations
2. Using NumPy's efficient reshape and transpose operations
3. Validating patterns before performing operations to catch errors early

## Future Improvements

Potential areas for enhancement include:
1. Adding support for repeating axes
2. Implementing non-adjacent axis merging
3. Extending to support other frameworks like PyTorch and TensorFlow
4. Optimizing parsing for frequently used patterns




