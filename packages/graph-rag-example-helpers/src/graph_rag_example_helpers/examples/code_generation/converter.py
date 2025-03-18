import ast
import json
import os
import typing
from textwrap import indent
from typing import Any

import griffe


def convert(
    package_name: str,
    search_paths: list[str],
    docstring_parser: griffe.DocstringStyle,
    output_path: str,
) -> None:
    """
    Load and convert a package's objects and documentation into a JSONL file.

    This method converts the internal documentation of modules, classes, functions, and
    attributes of a package into a format that is better suited for RAG (and GraphRAG
    in particular).

    The code uses the `griffe` library, which is a Python code analysis tool that
    extracts information from Python code and docstrings.

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


    Parameters
    ----------
    package_name :
        The name of the package to convert.
    search_paths :
        The paths to search for the package.
    docstring_parser :
        The docstring parser to use.
    output_path :
        The path to save the JSONL file.


    Examples
    --------
    from graph_rag_example_helpers.examples.code_generation.converter import convert
    convert("astrapy", [".venv/lib/python3.12/site-packages"], "google", "data")


    Notes
    -----
    - This code was written the `code-generation` example and `astrapy==1.5.2`. It will
      probably need tweaking for use with other python packages. Use at your own risk.
    """
    my_package = griffe.load(
        package_name, search_paths=search_paths, docstring_parser=docstring_parser
    )

    converter = _Converter()
    items = converter._convert(package_name, my_package)

    with open(os.path.join(output_path, f"{package_name}.jsonl"), "w") as f:
        for item in items:
            text = item.pop("text", "")
            id = item.get("path")
            metadata = item
            for key, value in metadata.items():
                if isinstance(value, set):
                    metadata[key] = list(value)
            f.write(json.dumps({"id": id, "text": text, "metadata": metadata}))
            f.write("\n")


class _Converter:
    """Converts griffe objects into ."""

    def __init__(self) -> None:
        self._alias_lookup: dict[str, str] = {}
        self._items: dict[str, dict[str, Any]] = {}
        self._bases: dict[str, set[str]] = {}
        self._used: bool = False

        self._typing_type_names: set[str] = {
            name for name in dir(typing) if not name.startswith("_")
        }
        """All standard-type names from the typing module."""

    def _check_first_use(self):
        assert not self._used, "Converters cannot be re-used."
        self._used = True

    def _extract_alias(self, name, alias: griffe.Alias):
        try:
            self._alias_lookup[name] = alias.final_target.path
        except Exception:
            pass

    def _is_first_letter_lowercase(self, s: str) -> bool:
        return s[:1].islower()

    def _extract_expr(
        self, annotation: str | griffe.Expr | None, gathered_types: set[str]
    ) -> str | None:
        if isinstance(annotation, griffe.Expr):
            annotation = annotation.modernize()
            for expr in annotation.iterate(flat=True):
                if isinstance(expr, griffe.ExprName):
                    if (
                        self._is_first_letter_lowercase(expr.name)
                        or expr.name in self._typing_type_names
                    ):
                        continue
                    gathered_types.add(expr.name)
            return annotation.__str__()
        else:
            return annotation

    def _extract_named_element(
        self,
        el: griffe.DocstringNamedElement,
        gathered_types: set[str],
        annotation_key: str = "type",
    ):
        stuff = {}
        if el.name != "":
            stuff["name"] = el.name
        anno = self._extract_expr(el.annotation, gathered_types=gathered_types)
        if anno is not None:
            stuff[annotation_key] = anno
        if el.description != "":
            stuff["description"] = el.description
        if el.value is not None:
            value = self._extract_expr(el.value, gathered_types=gathered_types)
            if value is not None:
                stuff["value"] = value
        return stuff

    def _format_parameter(self, el: dict[str, str]) -> str:
        text = el["name"]
        if "value" in el and "default" in el:
            assert el["value"] == el["default"]

        if "type" in el:
            text += f": {el['type']}"
        if "default" in el:
            text += f" = {el['default']}"
        if "description" in el:
            desc = indent(el["description"], "\t")
            text += f"\n\t{desc}"
        return text

    def _format_return(self, el: dict[str, str]) -> str:
        items = []
        if "type" in el:
            items.append(el["type"])
        if "description" in el:
            items.append(indent(el["description"], "\t"))
        return "\n\t".join(items)

    def _extract_common(self, obj: griffe.Object) -> dict[str, Any]:
        common: dict[str, Any] = {
            "kind": obj.kind.value,
            "name": obj.name,
            "path": obj.path,
        }
        if len(obj.imports) > 0:
            common["imports"] = obj.imports

        exports = obj.exports
        if isinstance(exports, set):
            common["exports"] = list(exports)
        elif isinstance(exports, list):
            common["exports"] = []
            for export in exports:
                if isinstance(export, str):
                    common["exports"].append(export)
                elif isinstance(export, griffe.ExprName):
                    common["exports"].append(export.name)

        return common

    def _extract_module(
        self, obj: griffe.Module, gathered_types: set[str]
    ) -> dict[str, Any]:
        item = self._extract_common(obj)
        item["properties"] = {
            "is_init_module": obj.is_init_module,
            "is_package": obj.is_package,
            "is_subpackage": obj.is_subpackage,
            "is_namespace_package": obj.is_namespace_package,
            "is_namespace_subpackage": obj.is_namespace_subpackage,
        }
        if obj.is_init_module:
            for export in item.get("exports", []):
                if export in item["imports"]:
                    # add exported items to alias lookup so references can be found
                    self._alias_lookup[f"{item['path']}.{export}"] = item["imports"][
                        export
                    ]
        return item

    def _extract_class(
        self, obj: griffe.Class, gathered_types: set[str]
    ) -> dict[str, Any]:
        item = self._extract_common(obj)
        params = []
        for param in obj.parameters:
            if param.name == "self":
                continue
            el = {"name": param.name}
            default = self._extract_expr(param.default, gathered_types=gathered_types)
            if default is not None:
                el["default"] = default
            annotation = self._extract_expr(
                param.annotation, gathered_types=gathered_types
            )
            if annotation is not None:
                el["type"] = annotation
            params.append(el)
        if len(params) > 0:
            item["parameters"] = params

        bases = [
            self._extract_expr(b, gathered_types=gathered_types) for b in obj.bases
        ]
        if len(bases) > 0:
            item["bases"] = bases
        return item

    def _extract_function(
        self, obj: griffe.Function, gathered_types: set[str]
    ) -> dict[str, Any]:
        item = self._extract_common(obj)
        params = []
        for param in obj.parameters:
            if param.name == "self":
                continue
            el = {"name": param.name}
            default = self._extract_expr(param.default, gathered_types=gathered_types)
            if default is not None:
                el["default"] = default
            annotation = self._extract_expr(
                param.annotation, gathered_types=gathered_types
            )
            if annotation is not None:
                el["type"] = annotation
            params.append(el)
        if len(params) > 0:
            item["parameters"] = params

        item["returns"] = [
            {"type": self._extract_expr(obj.returns, gathered_types=gathered_types)}
        ]
        return item

    def _extract_attribute(
        self, obj: griffe.Attribute, gathered_types: set[str]
    ) -> dict[str, Any]:
        item = self._extract_common(obj)
        el = {"name": obj.name}
        value = self._extract_expr(obj.value, gathered_types=gathered_types)
        if value is not None:
            el["default"] = value
        annotation = self._extract_expr(obj.annotation, gathered_types=gathered_types)
        if annotation is not None:
            el["type"] = annotation
        item["value"] = self._format_parameter(el)
        return item

    def _extract_object(self, name, obj: griffe.Object):
        assert name == obj.name

        if not obj.name.startswith("_"):
            gathered_types: set[str] = set()
            references: set[str] = set()
            if isinstance(obj, griffe.Attribute):
                item = self._extract_attribute(obj, gathered_types=gathered_types)
            elif isinstance(obj, griffe.Function):
                item = self._extract_function(obj, gathered_types=gathered_types)
            elif isinstance(obj, griffe.Class):
                item = self._extract_class(obj, gathered_types=gathered_types)
            elif isinstance(obj, griffe.Module):
                item = self._extract_module(obj, gathered_types=gathered_types)
            else:
                raise TypeError(f"Unknown obj type: {obj}")

            if obj.docstring is not None:
                for section in obj.docstring.parsed:
                    # TODO: merge this stuff with those from above if already existing.
                    if isinstance(section, griffe.DocstringSectionText):
                        item["text"] = section.value
                    elif isinstance(section, griffe.DocstringSectionAdmonition):
                        admonition_label = self._extract_expr(
                            section.value.annotation, gathered_types=gathered_types
                        )
                        if admonition_label is not None:
                            item[admonition_label] = section.value.description
                            if admonition_label == "example":
                                references.update(
                                    self._extract_imported_objects(
                                        section.value.description
                                    )
                                )
                    elif isinstance(section, griffe.DocstringSectionParameters):
                        params = []
                        for param in section.value:
                            named_element = self._extract_named_element(
                                param, gathered_types=gathered_types
                            )
                            named_element["default"] = self._extract_expr(
                                param.default, gathered_types=gathered_types
                            )
                            params.append(named_element)
                        item["parameters"] = params
                    elif isinstance(section, griffe.DocstringSectionAttributes):
                        item["attributes"] = [
                            self._extract_named_element(
                                e, gathered_types=gathered_types
                            )
                            for e in section.value
                        ]
                    elif isinstance(section, griffe.DocstringSectionYields):
                        item["yields"] = [
                            self._extract_named_element(
                                e, gathered_types=gathered_types
                            )
                            for e in section.value
                        ]
                    elif isinstance(section, griffe.DocstringSectionReturns):
                        item["returns"] = [
                            self._extract_named_element(
                                e, gathered_types=gathered_types
                            )
                            for e in section.value
                        ]
                    elif isinstance(section, griffe.DocstringSectionExamples):
                        for example in section.value:
                            references.update(
                                self._extract_imported_objects(example[1])
                            )
                        item["example"] = "/n/n/n".join(
                            [example[1] for example in section.value]
                        )
                    else:
                        raise TypeError(
                            f"Unknown section type: {section} of kind: {section.kind}"
                        )

            if item["path"] in references:
                references.remove(item["path"])
            if len(references) > 0:
                item["references"] = list(references)

            if len(gathered_types) > 0:
                item["gathered_types"] = list(gathered_types)

            if obj.path in self._items:
                raise Exception(f"{obj.path} was already found")
            self._items[obj.path] = item

        for _name, _obj in obj.members.items():
            self._extract_object_or_alias(_name, _obj)

    def _extract_object_or_alias(self, name: str, obj: griffe.Object | griffe.Alias):
        if isinstance(obj, griffe.Object):
            self._extract_object(name, obj)
        elif isinstance(obj, griffe.Alias):
            self._extract_alias(name, obj)

    def _extract_imported_objects(self, code: str) -> set[str]:
        """
        Extract the fully qualified names of imported objects from a given code snippet.

        If an error occurs, it removes the code from the error and beyond, and retries.

        Returns
        -------
        The set of imported types
        """
        code = (
            code.replace("\n>>>", "\n")
            .replace("\n...", "\n")
            .replace(">>> ", "")
            .replace("\n ", "\n")
            .replace("imort", "import")
        )
        imported_objects = set()

        while code:
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        for alias in node.names:
                            imported_objects.add(f"{node.module}.{alias.name}")
                break  # Stop retrying if parsing succeeds
            except SyntaxError as e:
                # Trim code before the error line and retry
                error_line = e.lineno
                if error_line is None or error_line <= 1:
                    break  # If error is at the first line, there's nothing to salvage
                code = "\n".join(code.splitlines()[: error_line - 1])

        return imported_objects

    def _update_item_paths(self, item: dict[str, Any]):
        """Update types to full paths for item attributes."""
        for key in ["gathered_types", "bases", "references"]:
            if key in item:
                updated = [self._alias_lookup.get(k, k) for k in item[key]]
                item[key] = updated

        if "bases" in item:
            for base in item["bases"]:
                self._bases.setdefault(base, set()).add(item["path"])

    def _convert(
        self, package_name: str, package: griffe.Object | griffe.Alias
    ) -> list[dict[str, Any]]:
        self._check_first_use()
        self._extract_object_or_alias(package_name, package)

        for item in self._items.values():
            self._update_item_paths(item)

        for base, implemented_by in self._bases.items():
            if base in self._items:
                self._items[base]["implemented_by"] = implemented_by

        return list(self._items.values())
