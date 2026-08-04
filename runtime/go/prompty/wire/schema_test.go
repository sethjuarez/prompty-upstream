package wire_test

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	model "prompty/model"
	wire "prompty/wire"
)

// prop builds a PropertySpec from the normalised map spelling, which is the
// same shape the runtime hydrates emitted properties from.
func prop(t *testing.T, m map[string]interface{}) wire.PropertySpec {
	t.Helper()

	spec, ok := wire.InspectProperty(m)
	if !ok {
		t.Fatalf("InspectProperty rejected %v", m)
	}
	return spec
}

func assertSchema(t *testing.T, label string, got interface{}, wantJSON string) {
	t.Helper()

	var want interface{}
	if err := json.Unmarshal([]byte(wantJSON), &want); err != nil {
		t.Fatalf("%s: bad expectation JSON: %v", label, err)
	}

	encoded, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("%s: marshal: %v", label, err)
	}
	var normalized interface{}
	if err := json.Unmarshal(encoded, &normalized); err != nil {
		t.Fatalf("%s: unmarshal: %v", label, err)
	}

	if !reflect.DeepEqual(normalized, want) {
		gotPretty, _ := json.MarshalIndent(normalized, "", "  ")
		wantPretty, _ := json.MarshalIndent(want, "", "  ")
		t.Errorf("%s mismatch\n--- got ---\n%s\n--- want ---\n%s", label, gotPretty, wantPretty)
	}
}

// ---------------------------------------------------------------------------
// Kind mapping
// ---------------------------------------------------------------------------

func TestKindToJSONType(t *testing.T) {
	cases := map[string]string{
		"string": "string", "integer": "integer", "float": "number",
		"number": "number", "boolean": "boolean", "array": "array", "object": "object",
	}
	for kind, want := range cases {
		got, ok := wire.KindToJSONType(kind)
		if !ok || got != want {
			t.Errorf("KindToJSONType(%q) = %q, %v; want %q, true", kind, got, ok, want)
		}
	}
	for _, kind := range []string{"union", "", "thread", "image"} {
		if got, ok := wire.KindToJSONType(kind); ok {
			t.Errorf("KindToJSONType(%q) = %q, true; want no type keyword", kind, got)
		}
	}
}

// ---------------------------------------------------------------------------
// Nested objects and arrays
// ---------------------------------------------------------------------------

func TestNestedObjectSchema(t *testing.T) {
	spec := prop(t, map[string]interface{}{
		"name": "address", "kind": "object", "description": "A postal address",
		"properties": []interface{}{
			map[string]interface{}{"name": "street", "kind": "string", "required": true},
			map[string]interface{}{"name": "zip", "kind": "string"},
		},
	})

	schema, err := wire.PropertySchema(spec, false)
	if err != nil {
		t.Fatalf("PropertySchema: %v", err)
	}
	assertSchema(t, "nested object", schema, `{
		"type": "object",
		"description": "A postal address",
		"properties": {"street": {"type": "string"}, "zip": {"type": "string"}},
		"required": ["street"],
		"additionalProperties": false
	}`)
}

// TestNestedObjectSchemaStrict shows the strict dialect widening the optional
// field instead of leaving it out of `required`.
func TestNestedObjectSchemaStrict(t *testing.T) {
	spec := prop(t, map[string]interface{}{
		"name": "address", "kind": "object",
		"properties": []interface{}{
			map[string]interface{}{"name": "street", "kind": "string", "required": true},
			map[string]interface{}{"name": "zip", "kind": "string"},
		},
	})

	schema, err := wire.PropertySchema(spec, true)
	if err != nil {
		t.Fatalf("PropertySchema: %v", err)
	}
	assertSchema(t, "nested object strict", schema, `{
		"type": "object",
		"properties": {"street": {"type": "string"}, "zip": {"type": ["string", "null"]}},
		"required": ["street"],
		"additionalProperties": false
	}`)
}

func TestArrayOfObjectsSchema(t *testing.T) {
	spec := prop(t, map[string]interface{}{
		"name": "items", "kind": "array",
		"items": map[string]interface{}{
			"kind": "object",
			"properties": []interface{}{
				map[string]interface{}{"name": "sku", "kind": "string", "required": true},
				map[string]interface{}{"name": "qty", "kind": "integer", "required": true},
			},
		},
	})

	schema, err := wire.PropertySchema(spec, false)
	if err != nil {
		t.Fatalf("PropertySchema: %v", err)
	}
	assertSchema(t, "array of objects", schema, `{
		"type": "array",
		"items": {
			"type": "object",
			"properties": {"sku": {"type": "string"}, "qty": {"type": "integer"}},
			"required": ["sku", "qty"],
			"additionalProperties": false
		}
	}`)
}

// TestBareCompositeSchemas proves an unspecified element or field set produces
// a permissive schema rather than an error.
func TestBareCompositeSchemas(t *testing.T) {
	array, err := wire.PropertySchema(prop(t, map[string]interface{}{"name": "a", "kind": "array"}), false)
	if err != nil {
		t.Fatalf("array: %v", err)
	}
	assertSchema(t, "bare array", array, `{"type": "array"}`)

	object, err := wire.PropertySchema(prop(t, map[string]interface{}{"name": "o", "kind": "object"}), false)
	if err != nil {
		t.Fatalf("object: %v", err)
	}
	assertSchema(t, "bare object", object, `{"type": "object"}`)
}

func TestDeeplyNestedSchema(t *testing.T) {
	spec := prop(t, map[string]interface{}{
		"name": "root", "kind": "object",
		"properties": []interface{}{
			map[string]interface{}{
				"name": "level1", "kind": "object", "required": true,
				"properties": []interface{}{
					map[string]interface{}{
						"name": "level2", "kind": "array", "required": true,
						"items": map[string]interface{}{"kind": "integer"},
					},
				},
			},
		},
	})

	schema, err := wire.PropertySchema(spec, false)
	if err != nil {
		t.Fatalf("PropertySchema: %v", err)
	}
	assertSchema(t, "deep nesting", schema, `{
		"type": "object",
		"properties": {
			"level1": {
				"type": "object",
				"properties": {"level2": {"type": "array", "items": {"type": "integer"}}},
				"required": ["level2"],
				"additionalProperties": false
			}
		},
		"required": ["level1"],
		"additionalProperties": false
	}`)
}

// ---------------------------------------------------------------------------
// Unions
// ---------------------------------------------------------------------------

func TestUnionAnyOfSchema(t *testing.T) {
	spec := prop(t, map[string]interface{}{
		"name": "value", "kind": "union",
		"anyOf": []interface{}{
			map[string]interface{}{"kind": "string"},
			map[string]interface{}{"kind": "integer"},
		},
	})

	schema, err := wire.PropertySchema(spec, false)
	if err != nil {
		t.Fatalf("PropertySchema: %v", err)
	}
	assertSchema(t, "union anyOf", schema, `{"anyOf": [{"type": "string"}, {"type": "integer"}]}`)
}

// TestUnionNullableAddsNullBranch shows a nullable union gaining a null branch
// rather than a two-element type array, which would be invalid beside anyOf.
func TestUnionNullableAddsNullBranch(t *testing.T) {
	spec := prop(t, map[string]interface{}{
		"name": "value", "kind": "union", "nullable": true,
		"anyOf": []interface{}{
			map[string]interface{}{"kind": "string"},
			map[string]interface{}{"kind": "boolean"},
		},
	})

	schema, err := wire.PropertySchema(spec, false)
	if err != nil {
		t.Fatalf("PropertySchema: %v", err)
	}
	assertSchema(t, "nullable union", schema,
		`{"anyOf": [{"type": "string"}, {"type": "boolean"}, {"type": "null"}]}`)
}

// TestUnionOneOfIsRejected: neither provider's structured-output subset
// implements exactly-one-match validation, so silently downgrading oneOf to
// anyOf would change the contract the author asked for.
func TestUnionOneOfIsRejected(t *testing.T) {
	spec := prop(t, map[string]interface{}{
		"name": "value", "kind": "union",
		"oneOf": []interface{}{
			map[string]interface{}{"kind": "string"},
			map[string]interface{}{"kind": "integer"},
		},
	})

	_, err := wire.PropertySchema(spec, false)
	if err == nil {
		t.Fatal("expected oneOf to be rejected")
	}
	if !errors.Is(err, wire.ErrSchema) {
		t.Errorf("error does not wrap wire.ErrSchema: %v", err)
	}
}

func TestUnionWithoutBranchesIsRejected(t *testing.T) {
	spec := prop(t, map[string]interface{}{"name": "value", "kind": "union"})
	if _, err := wire.PropertySchema(spec, false); err == nil {
		t.Fatal("expected an empty union to be rejected")
	}
}

func TestUnionWithBothCompositionsIsRejected(t *testing.T) {
	spec := prop(t, map[string]interface{}{
		"name": "value", "kind": "union",
		"oneOf": []interface{}{map[string]interface{}{"kind": "string"}},
		"anyOf": []interface{}{map[string]interface{}{"kind": "integer"}},
	})
	if _, err := wire.PropertySchema(spec, false); err == nil {
		t.Fatal("expected a union with both compositions to be rejected")
	}
}

// TestNestedUnionErrorPropagates proves an invalid branch deep in a tree is not
// swallowed on the way out.
func TestNestedUnionErrorPropagates(t *testing.T) {
	spec := prop(t, map[string]interface{}{
		"name": "root", "kind": "object",
		"properties": []interface{}{
			map[string]interface{}{
				"name": "bad", "kind": "union",
				"oneOf": []interface{}{map[string]interface{}{"kind": "string"}},
			},
		},
	})
	if _, err := wire.PropertySchema(spec, false); err == nil {
		t.Fatal("expected the nested oneOf to be rejected")
	}
}

// ---------------------------------------------------------------------------
// Enums and nullability
// ---------------------------------------------------------------------------

func TestEnumSchema(t *testing.T) {
	spec := prop(t, map[string]interface{}{
		"name": "unit", "kind": "string",
		"enumValues": []interface{}{"celsius", "fahrenheit"},
	})

	schema, err := wire.PropertySchema(spec, false)
	if err != nil {
		t.Fatalf("PropertySchema: %v", err)
	}
	assertSchema(t, "enum", schema, `{"type": "string", "enum": ["celsius", "fahrenheit"]}`)
}

// TestNullableEnumAdmitsNull: widening the type to allow null while leaving the
// enum closed would produce a schema no value can satisfy.
func TestNullableEnumAdmitsNull(t *testing.T) {
	spec := prop(t, map[string]interface{}{
		"name": "unit", "kind": "string", "nullable": true,
		"enumValues": []interface{}{"celsius", "fahrenheit"},
	})

	schema, err := wire.PropertySchema(spec, false)
	if err != nil {
		t.Fatalf("PropertySchema: %v", err)
	}
	assertSchema(t, "nullable enum", schema,
		`{"type": ["string", "null"], "enum": ["celsius", "fahrenheit", null]}`)
}

// TestNullableIsNotDoubleApplied guards the strict path: a property already
// declared nullable must not be widened twice.
func TestNullableIsNotDoubleApplied(t *testing.T) {
	schema, err := wire.ParametersSchema([]wire.PropertySpec{
		prop(t, map[string]interface{}{"name": "note", "kind": "string", "nullable": true}),
	}, true)
	if err != nil {
		t.Fatalf("ParametersSchema: %v", err)
	}
	assertSchema(t, "already nullable", schema, `{
		"type": "object",
		"properties": {"note": {"type": ["string", "null"]}},
		"required": ["note"]
	}`)
}

// ---------------------------------------------------------------------------
// Parameters and outputs
// ---------------------------------------------------------------------------

func TestParametersSchemaRequiredSemantics(t *testing.T) {
	params := []wire.PropertySpec{
		prop(t, map[string]interface{}{"name": "a", "kind": "string", "required": true}),
		prop(t, map[string]interface{}{"name": "b", "kind": "integer"}),
	}

	lenient, err := wire.ParametersSchema(params, false)
	if err != nil {
		t.Fatalf("lenient: %v", err)
	}
	assertSchema(t, "lenient parameters", lenient, `{
		"type": "object",
		"properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
		"required": ["a"]
	}`)

	strict, err := wire.ParametersSchema(params, true)
	if err != nil {
		t.Fatalf("strict: %v", err)
	}
	assertSchema(t, "strict parameters", strict, `{
		"type": "object",
		"properties": {"a": {"type": "string"}, "b": {"type": ["integer", "null"]}},
		"required": ["a", "b"]
	}`)
}

// TestParametersSchemaRequiredOrderIsStable: `required` is a JSON array, so a
// builder that derived it from map iteration would produce a different request
// body on every call.
func TestParametersSchemaRequiredOrderIsStable(t *testing.T) {
	params := []wire.PropertySpec{
		prop(t, map[string]interface{}{"name": "zulu", "kind": "string", "required": true}),
		prop(t, map[string]interface{}{"name": "alpha", "kind": "string", "required": true}),
		prop(t, map[string]interface{}{"name": "mike", "kind": "string", "required": true}),
	}

	first, err := wire.ParametersSchema(params, false)
	if err != nil {
		t.Fatalf("ParametersSchema: %v", err)
	}
	want := first["required"]
	for i := 0; i < 32; i++ {
		again, err := wire.ParametersSchema(params, false)
		if err != nil {
			t.Fatalf("repeat %d: %v", i, err)
		}
		if !reflect.DeepEqual(again["required"], want) {
			t.Fatalf("required order is unstable: %v then %v", want, again["required"])
		}
	}
	assertSchema(t, "declaration order", want, `["zulu", "alpha", "mike"]`)
}

func TestOutputsSchemaIsAlwaysStrict(t *testing.T) {
	outputs := []wire.PropertySpec{
		prop(t, map[string]interface{}{"name": "city", "kind": "string", "required": true}),
		prop(t, map[string]interface{}{"name": "note", "kind": "string"}),
	}

	schema, err := wire.OutputsSchema(outputs)
	if err != nil {
		t.Fatalf("OutputsSchema: %v", err)
	}
	assertSchema(t, "outputs", schema, `{
		"type": "object",
		"properties": {"city": {"type": "string"}, "note": {"type": ["string", "null"]}},
		"required": ["city", "note"],
		"additionalProperties": false
	}`)
}

func TestOutputsSchemaEmptyIsNil(t *testing.T) {
	schema, err := wire.OutputsSchema(nil)
	if err != nil || schema != nil {
		t.Errorf("OutputsSchema(nil) = %v, %v; want nil, nil", schema, err)
	}
}

// TestAnthropicOutputConfigUsesThePlainDialect documents the deliberate
// difference from OpenAI: Anthropic omits an optional output from `required`
// instead of making it nullable.
func TestAnthropicOutputConfigUsesThePlainDialect(t *testing.T) {
	outputs := []wire.PropertySpec{
		prop(t, map[string]interface{}{"name": "city", "kind": "string", "required": true}),
		prop(t, map[string]interface{}{"name": "note", "kind": "string"}),
	}

	config, err := wire.AnthropicOutputConfig(outputs)
	if err != nil {
		t.Fatalf("AnthropicOutputConfig: %v", err)
	}
	assertSchema(t, "anthropic output_config", config, `{
		"format": {
			"type": "json_schema",
			"schema": {
				"type": "object",
				"properties": {"city": {"type": "string"}, "note": {"type": "string"}},
				"required": ["city"],
				"additionalProperties": false
			}
		}
	}`)
}

// ---------------------------------------------------------------------------
// Property inspection over emitted values
// ---------------------------------------------------------------------------

// TestInspectEmittedPropertyValues proves the projection reads the emitted
// structs, not just the map spelling — including through pointers.
func TestInspectEmittedPropertyValues(t *testing.T) {
	required := true
	description := "A city name"

	spec, ok := wire.InspectProperty(model.Property{
		Name: "city", Kind: "string", Required: &required, Description: &description,
	})
	if !ok {
		t.Fatal("InspectProperty rejected an emitted Property")
	}
	if spec.Name != "city" || spec.Kind != "string" || !spec.Required || spec.Description != description {
		t.Errorf("spec = %+v", spec)
	}

	nested := model.ObjectProperty{
		Name: "address", Kind: "object",
		Properties: []interface{}{
			model.Property{Name: "street", Kind: "string", Required: &required},
		},
	}
	objectSpec, ok := wire.InspectProperty(&nested)
	if !ok {
		t.Fatal("InspectProperty rejected an emitted *ObjectProperty")
	}
	if len(objectSpec.Properties) != 1 || objectSpec.Properties[0].Name != "street" {
		t.Errorf("nested properties = %+v", objectSpec.Properties)
	}

	arraySpec, ok := wire.InspectProperty(model.ArrayProperty{
		Name: "tags", Kind: "array", Items: model.Property{Kind: "string"},
	})
	if !ok {
		t.Fatal("InspectProperty rejected an emitted ArrayProperty")
	}
	if arraySpec.Items == nil || arraySpec.Items.Kind != "string" {
		t.Errorf("array items = %+v", arraySpec.Items)
	}
}

// TestInspectPropertyRejectsNonProperties: a nil or foreign value must be
// reported, never panicked on.
func TestInspectPropertyRejectsNonProperties(t *testing.T) {
	for _, bad := range []interface{}{
		nil, 42, "string", []interface{}{},
		(*model.Property)(nil), (*model.ObjectProperty)(nil),
		(*model.ArrayProperty)(nil), (*model.UnionProperty)(nil),
	} {
		if _, ok := wire.InspectProperty(bad); ok {
			t.Errorf("InspectProperty(%#v) reported success", bad)
		}
	}
}

// TestInspectPropertySurvivesTheGeneratedLoaderDefect documents the known
// generated baseline bug: model.LoadProperty drops every field for a primitive
// kind. Inspection must not crash on the resulting zero value, and callers must
// be able to see that it carries nothing.
func TestInspectPropertySurvivesTheGeneratedLoaderDefect(t *testing.T) {
	loaded, err := model.LoadProperty(
		map[string]interface{}{"name": "city", "kind": "string", "required": true},
		model.NewLoadContext())
	if err != nil {
		t.Fatalf("LoadProperty: %v", err)
	}

	spec, ok := wire.InspectProperty(loaded)
	if !ok {
		t.Fatal("InspectProperty rejected the emitted loader's output")
	}
	if spec.Name != "" || spec.Kind != "" {
		t.Skip("the generated primitive Property load defect appears to be fixed; " +
			"this test documents the workaround and can be retired")
	}
}
