# AstraPy Documentation

This data file contains the AstraPy documentation in a specialized format for use
in the GraphRAG `code_generation` example.

## Generation

The file was generated using `astrapy` version `1.5.2` via the `convert` method in
`graph_rag_example_helpers.examples.code_generation.converter`. See the help on the
method for more information about how to use it.

## Structure

The JSONL file contains one JSON object per line, with the following structure:

id: the path to the object in the package
text: the description of the object (if any, can be empty)

metadata: Always includes `name`, `path`, `kind` keys.
            The remaining keys below are included when available.
    name: the name of the object
    path: the path to the object in the package
    kind: either `module`, `class`, `function`, or `attribute`
    parameters: the parameters for a class or function. Includes type
        information, default values, and descriptions
    attributes: the attributes on a class or module. Includes type
        information and descriptions
    gathered_types: list of non-standard types in the parameters and attributes
    imports: list of non-standard types imported by the class or module
    exports: list of non-standard types exported by the module
    properties: list of boolean properties about the module
    example: any code examples for the class, function, or module
    references: list of any non-standard types used in the example code
    returns: the return type and description
    yields: the yield type and description
    bases: list of base types inherited by the class
    implemented_by: list of types that implement the a base class
