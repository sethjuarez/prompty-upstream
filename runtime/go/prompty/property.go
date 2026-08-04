package prompty

import (
	model "prompty/model"
)

// hydrateProperty builds a fully populated emitted property value from a
// normalised property map.
//
// Why this exists: the emitted model.LoadProperty dispatches on `kind` and only
// has cases for "array", "object" and "union"; every primitive kind falls
// through to `return result, nil` with a zero-valued Property, dropping name,
// kind, default and description. That is a known defect in the generated code,
// and generated files are not editable here, so the runtime hydrates properties
// itself from the same normalised maps it hands to the emitter. The emitted
// types stay canonical — only the population path is ours.
func hydrateProperty(data interface{}) (interface{}, error) {
	m, ok := data.(map[string]interface{})
	if !ok {
		return nil, newValueError("Invalid property: expected a mapping")
	}

	base := model.Property{
		Name:        stringField(m, "name"),
		Kind:        stringField(m, "kind"),
		Description: optionalStringField(m, "description"),
		Required:    optionalBoolField(m, "required"),
		Nullable:    optionalBoolField(m, "nullable"),
		Default:     optionalAnyField(m, "default"),
		Example:     optionalAnyField(m, "example"),
		EnumValues:  anySliceField(m, "enumValues"),
	}

	switch base.Kind {
	case "array":
		out := model.ArrayProperty{
			Name: base.Name, Kind: base.Kind, Description: base.Description,
			Required: base.Required, Nullable: base.Nullable, Default: base.Default,
			Example: base.Example, EnumValues: base.EnumValues,
		}
		if items, ok := m["items"]; ok && items != nil {
			child, err := hydrateProperty(items)
			if err != nil {
				return nil, err
			}
			out.Items = child
		}
		return out, nil

	case "object":
		out := model.ObjectProperty{
			Name: base.Name, Kind: base.Kind, Description: base.Description,
			Required: base.Required, Nullable: base.Nullable, Default: base.Default,
			Example: base.Example, EnumValues: base.EnumValues,
		}
		children, err := hydrateProperties(m["properties"])
		if err != nil {
			return nil, err
		}
		out.Properties = children
		return out, nil

	case "union":
		out := model.UnionProperty{
			Name: base.Name, Kind: base.Kind, Description: base.Description,
			Required: base.Required, Nullable: base.Nullable, Default: base.Default,
			Example: base.Example, EnumValues: base.EnumValues,
		}
		oneOf, err := hydrateProperties(m["oneOf"])
		if err != nil {
			return nil, err
		}
		anyOf, err := hydrateProperties(m["anyOf"])
		if err != nil {
			return nil, err
		}
		out.OneOf, out.AnyOf = oneOf, anyOf
		return out, nil

	default:
		return base, nil
	}
}

func hydrateProperties(v interface{}) ([]interface{}, error) {
	if v == nil {
		return nil, nil
	}
	list, ok := v.([]interface{})
	if !ok {
		return nil, newValueError("Invalid property list: expected a sequence")
	}
	out := make([]interface{}, 0, len(list))
	for _, item := range list {
		prop, err := hydrateProperty(item)
		if err != nil {
			return nil, err
		}
		out = append(out, prop)
	}
	return out, nil
}

// PropertyView is a read-only projection over the emitted property variants.
// Callers need name/kind/required/default without caring which concrete emitted
// struct they are looking at.
type PropertyView struct {
	Name     string
	Kind     string
	Required bool
	Default  *interface{}
}

// viewProperty projects any emitted property variant (value or pointer) into a
// PropertyView. The second result is false for values that are not properties.
func viewProperty(v interface{}) (PropertyView, bool) {
	switch p := v.(type) {
	case model.Property:
		return newPropertyView(p.Name, p.Kind, p.Required, p.Default), true
	case *model.Property:
		return newPropertyView(p.Name, p.Kind, p.Required, p.Default), true
	case model.ArrayProperty:
		return newPropertyView(p.Name, p.Kind, p.Required, p.Default), true
	case *model.ArrayProperty:
		return newPropertyView(p.Name, p.Kind, p.Required, p.Default), true
	case model.ObjectProperty:
		return newPropertyView(p.Name, p.Kind, p.Required, p.Default), true
	case *model.ObjectProperty:
		return newPropertyView(p.Name, p.Kind, p.Required, p.Default), true
	case model.UnionProperty:
		return newPropertyView(p.Name, p.Kind, p.Required, p.Default), true
	case *model.UnionProperty:
		return newPropertyView(p.Name, p.Kind, p.Required, p.Default), true
	default:
		return PropertyView{}, false
	}
}

func newPropertyView(name, kind string, required *bool, def *interface{}) PropertyView {
	view := PropertyView{Name: name, Kind: kind, Default: def}
	if required != nil {
		view.Required = *required
	}
	return view
}

func stringField(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func optionalStringField(m map[string]interface{}, key string) *string {
	if v, ok := m[key].(string); ok {
		return &v
	}
	return nil
}

func optionalBoolField(m map[string]interface{}, key string) *bool {
	if v, ok := m[key].(bool); ok {
		return &v
	}
	return nil
}

func optionalAnyField(m map[string]interface{}, key string) *interface{} {
	v, ok := m[key]
	if !ok || v == nil {
		return nil
	}
	return &v
}

func anySliceField(m map[string]interface{}, key string) []interface{} {
	if v, ok := m[key].([]interface{}); ok {
		return v
	}
	return nil
}
