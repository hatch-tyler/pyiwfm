Contributing
============

We welcome contributions to pyiwfm! This guide explains how to contribute.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

       git clone https://github.com/YOUR_USERNAME/pyiwfm.git
       cd pyiwfm

3. Create a virtual environment and install development dependencies:

   .. code-block:: bash

       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate
       pip install -e ".[dev]"

4. Create a branch for your changes:

   .. code-block:: bash

       git checkout -b feature/your-feature-name

Development Workflow
--------------------

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

    # Run all tests
    pytest tests/

    # Run with coverage
    pytest tests/ --cov=pyiwfm --cov-report=html

    # Run specific test file
    pytest tests/unit/test_mesh.py -v

Code Style
~~~~~~~~~~

We use the following tools for code quality:

- **Ruff** for linting and formatting
- **mypy** for type checking

.. code-block:: bash

    # Format code
    ruff format src/ tests/

    # Lint code
    ruff check src/ tests/

    # Type check
    mypy src/pyiwfm/

Pre-commit Hooks
~~~~~~~~~~~~~~~~

Install pre-commit hooks to automatically check code:

.. code-block:: bash

    pip install pre-commit
    pre-commit install

Writing Tests
-------------

All new features should include tests. Follow these guidelines:

1. Place unit tests in ``tests/unit/``
2. Use pytest fixtures for common setup
3. Name test files ``test_*.py``
4. Name test functions ``test_*``

Example test:

.. code-block:: python

    import pytest
    from pyiwfm.core.mesh import Node

    def test_node_creation():
        node = Node(id=1, x=100.0, y=200.0)
        assert node.id == 1
        assert node.x == 100.0
        assert node.y == 200.0

Documentation
-------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd docs
    pip install -r requirements.txt
    make html

The built documentation will be in ``docs/_build/html/``.

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

- Use NumPy-style docstrings
- Include examples in docstrings
- Add new modules to the appropriate RST files

Docstring Example
~~~~~~~~~~~~~~~~~

.. code-block:: python

    def example_function(param1: int, param2: str = "default") -> bool:
        """
        Short description of the function.

        Longer description if needed, explaining the function's purpose
        and behavior in more detail.

        Parameters
        ----------
        param1 : int
            Description of param1.
        param2 : str, optional
            Description of param2. Default is "default".

        Returns
        -------
        bool
            Description of return value.

        Raises
        ------
        ValueError
            If param1 is negative.

        Examples
        --------
        >>> example_function(1)
        True
        >>> example_function(1, param2="test")
        True
        """
        pass

Submitting Changes
------------------

1. Ensure all tests pass
2. Update documentation if needed
3. Commit your changes with a descriptive message:

   .. code-block:: bash

       git commit -m "Add feature X that does Y"

4. Push to your fork:

   .. code-block:: bash

       git push origin feature/your-feature-name

5. Open a Pull Request on GitHub

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

- Provide a clear description of the changes
- Reference any related issues
- Include tests for new features
- Update documentation as needed
- Ensure CI passes

Code of Conduct
---------------

Please be respectful and constructive in all interactions. We follow the
`Contributor Covenant <https://www.contributor-covenant.org/>`_ code of conduct.

Questions?
----------

If you have questions, please:

1. Check existing issues and documentation
2. Open a new issue with a clear description
3. Contact the maintainers
