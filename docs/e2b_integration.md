# E2B Code Interpreter Integration

This document explains how to use the E2B code interpreter integration for secure code execution in Cortex agents.

## Overview

The E2B integration provides secure, isolated Python code execution for coding agents. It automatically creates sandboxed environments that are:
- **Secure**: Isolated from the host system
- **Temporary**: Automatically cleaned up after execution
- **Configurable**: Customizable timeout and package installation
- **Safe**: No access to sensitive system resources

## Backend Integration

### Automatic Tool Addition

When creating an agent with `agent_type: "coding"`, the system automatically adds E2B tools:

```python
# Example request payload
{
    "message": "Calculate the factorial of 10",
    "agent_type": "coding",
    "system_prompt": "You are a Python coding assistant...",
    "provider_id": "openai",
    "model_id": "gpt-4",
    "tools": []
}
```

### Available Tools

#### 1. `execute_python_code`
Executes Python code in a secure sandbox.

**Parameters:**
- `code` (str): Python code to execute
- `timeout` (int, optional): Sandbox timeout in seconds (default: 300)
- `install_packages` (List[str], optional): Packages to install before execution

**Example:**
```python
# The agent can call this tool with:
{
    "code": "import pandas as pd\nprint('Hello from E2B!')",
    "timeout": 300,
    "install_packages": ["pandas", "numpy"]
}
```

**Response:**
```json
{
    "success": true,
    "output": "Hello from E2B!",
    "sandbox_id": "abc123",
    "execution_time": 0.5,
    "packages_installed": ["pandas", "numpy"],
    "code": "import pandas as pd\nprint('Hello from E2B!')"
}
```

#### 2. `get_sandbox_info`
Retrieves information about sandbox sessions.

**Parameters:**
- `sandbox_id` (str, optional): Specific sandbox ID to query

## Frontend Integration

### Coding Agent Configuration

Use the `CodingAgentConfig` component to configure coding agents:

```tsx
import CodingAgentConfig from '@/components/ui/agents_ui/CodingAgentConfig';

function CreateCodingAgent() {
  const handleSave = (config) => {
    // Save the coding agent configuration
    console.log('Coding agent config:', config);
  };

  return (
    <CodingAgentConfig 
      onSave={handleSave}
      initialConfig={{
        name: "Data Analysis Assistant",
        defaultPackages: ["pandas", "numpy", "matplotlib"],
        timeout: 600
      }}
    />
  );
}
```

### Agent Creation Flow

1. **Select Agent Type**: Choose "Coding Agent" in the agent builder
2. **Configure Sandbox**: Set timeout, packages, and security settings
3. **System Prompt**: Define the agent's behavior and capabilities
4. **Deploy**: The agent automatically gets E2B tools

## Usage Examples

### Basic Code Execution

```python
# User: "Write a function to calculate fibonacci numbers"

# Agent automatically uses execute_python_code tool:
{
    "code": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the function
for i in range(10):
    print(f"fibonacci({i}) = {fibonacci(i)}")
""",
    "timeout": 300
}
```

### Data Analysis

```python
# User: "Analyze this dataset and create a visualization"

# Agent can install packages and execute:
{
    "code": """
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create sample data
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(data['x'], data['y'])
plt.title('Sample Data Visualization')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.show()
""",
    "install_packages": ["pandas", "matplotlib", "numpy"]
}
```

### Machine Learning

```python
# User: "Train a simple machine learning model"

# Agent executes with ML packages:
{
    "code": """
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.2f}")
""",
    "install_packages": ["scikit-learn"]
}
```

## Security Features

### Sandbox Isolation
- Each code execution runs in a completely isolated environment
- No access to host system files or network (unless explicitly enabled)
- Automatic cleanup after execution

### Timeout Protection
- Configurable execution time limits
- Default 5-minute timeout
- Automatic sandbox termination

### Package Management
- Controlled package installation
- Whitelist of allowed packages (configurable)
- Automatic package cleanup

### Resource Limits
- Memory and CPU limits
- File system size limits
- Network access controls

## Configuration Options

### Agent-Level Configuration

```typescript
interface CodingAgentConfig {
  name: string;
  description: string;
  systemPrompt: string;
  timeout: number;                    // Sandbox timeout in seconds
  defaultPackages: string[];          // Packages to install by default
  allowedPackages: string[];          // Whitelist of allowed packages
  maxExecutionTime: number;           // Max execution time per request
  enableFileSystem: boolean;          // Allow file system access
  enableNetwork: boolean;             // Allow network access
  enableGPU: boolean;                 // Allow GPU access
}
```

### Runtime Configuration

```python
# Example of runtime tool usage
{
    "code": "print('Hello World')",
    "timeout": 60,                    # 1 minute timeout
    "install_packages": ["requests"] # Install additional packages
}
```

## Error Handling

### Common Error Scenarios

1. **Timeout Errors**: Code execution exceeds timeout limit
2. **Package Installation Failures**: Invalid or unavailable packages
3. **Syntax Errors**: Invalid Python code
4. **Runtime Errors**: Code execution failures

### Error Response Format

```json
{
    "success": false,
    "error": "SyntaxError: invalid syntax",
    "code": "print('Hello World'",
    "sandbox_id": "abc123"
}
```

## Best Practices

### For Developers

1. **Always use the sandbox**: Never execute code directly on the server
2. **Set appropriate timeouts**: Balance functionality with security
3. **Limit package access**: Only allow necessary packages
4. **Monitor usage**: Track sandbox creation and execution

### For Users

1. **Be specific**: Provide clear instructions for code execution
2. **Include requirements**: Specify needed packages upfront
3. **Test incrementally**: Start with simple code and build complexity
4. **Handle errors**: Be prepared to debug and fix code issues

## Monitoring and Logging

### Sandbox Metrics

- Execution time tracking
- Package installation logs
- Error rate monitoring
- Resource usage statistics

### Logging

```python
# Backend logs include:
logger.info(f"Executing Python code in E2B sandbox: {code[:100]}...")
logger.info(f"Code execution completed successfully. Sandbox ID: {sandbox.sandbox_id}")
logger.error(f"Code execution error: {execution.error}")
```

## Troubleshooting

### Common Issues

1. **Sandbox Creation Failures**
   - Check E2B API key configuration
   - Verify network connectivity
   - Ensure sufficient resources

2. **Package Installation Issues**
   - Verify package names are correct
   - Check package availability
   - Review installation logs

3. **Timeout Issues**
   - Increase timeout for complex operations
   - Optimize code for better performance
   - Consider breaking down large operations

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
# Set log level to DEBUG
logger.setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features

1. **Multi-language Support**: JavaScript, R, Julia execution
2. **Persistent Sandboxes**: Long-running sandbox sessions
3. **File Management**: Upload/download files to/from sandbox
4. **Collaborative Sandboxes**: Shared sandbox environments
5. **Advanced Security**: Fine-grained access controls

### Integration Roadmap

1. **Phase 1**: Basic Python execution (Current)
2. **Phase 2**: Enhanced security and monitoring
3. **Phase 3**: Multi-language support
4. **Phase 4**: Advanced collaboration features

