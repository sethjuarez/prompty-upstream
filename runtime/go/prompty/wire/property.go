package wire

import (
	model "prompty/model"
)

// PropertySpec is a read-only, fully materialised projection of the emitted
// property variants (model.Property, ArrayProperty, ObjectProperty,
// UnionProperty) plus the raw normalised map form.
//
// Why a projection instead of the emitted types directly: the emitted
// model.LoadProperty dispatches on `kind` and only has cases for "array",
// "object" and "union" — every primitive kind falls through and returns a
// zero-valued Property, dropping name/kind/required. Generated files cannot be
// edited here, so consumers must never assume an emitted Property that arrived
// through that path is populated. Inspect* reads whatever shape is actually
// present, including the normalised maps the runtime hydrates from, and never
// panics on an unexpected value.
type PropertySpec struct {
	Name        string
	Kind        string
	Description string
	Required    bool
	Nullable    bool
	EnumValues  []interface{}

	// Items is the element schema of an array property, nil when unspecified.
	Items *PropertySpec
	// Properties are the fields of an object property.
	Properties []PropertySpec
	// OneOf and AnyOf are the branches of a union property. Exactly one of the
	// two is populated on a well-formed union.
	OneOf []PropertySpec
	AnyOf []PropertySpec
}

// InspectProperty projects any supported property representation into a
// PropertySpec. The second result is false for values that are not properties.
func InspectProperty(v interface{}) (PropertySpec, bool) {
	switch p := v.(type) {
	case model.Property:
		return baseSpec(p.Name, p.Kind, p.Description, p.Required, p.Nullable, p.EnumValues), true
	case *model.Property:
		if p == nil {
			return PropertySpec{}, false
		}
		return InspectProperty(*p)

	case model.ArrayProperty:
		spec := baseSpec(p.Name, p.Kind, p.Description, p.Required, p.Nullable, p.EnumValues)
		if p.Items != nil {
			if item, ok := InspectProperty(p.Items); ok {
				spec.Items = &item
			}
		}
		return spec, true
	case *model.ArrayProperty:
		if p == nil {
			return PropertySpec{}, false
		}
		return InspectProperty(*p)

	case model.ObjectProperty:
		spec := baseSpec(p.Name, p.Kind, p.Description, p.Required, p.Nullable, p.EnumValues)
		spec.Properties = InspectProperties(p.Properties)
		return spec, true
	case *model.ObjectProperty:
		if p == nil {
			return PropertySpec{}, false
		}
		return InspectProperty(*p)

	case model.UnionProperty:
		spec := baseSpec(p.Name, p.Kind, p.Description, p.Required, p.Nullable, p.EnumValues)
		spec.OneOf = InspectProperties(p.OneOf)
		spec.AnyOf = InspectProperties(p.AnyOf)
		return spec, true
	case *model.UnionProperty:
		if p == nil {
			return PropertySpec{}, false
		}
		return InspectProperty(*p)

	case PropertySpec:
		return p, true

	case map[string]interface{}:
		return inspectPropertyMap(p), true

	default:
		return PropertySpec{}, false
	}
}

// InspectProperties projects a heterogeneous emitted property list, skipping
// entries that are not properties.
func InspectProperties(list []interface{}) []PropertySpec {
	if len(list) == 0 {
		return nil
	}
	out := make([]PropertySpec, 0, len(list))
	for _, item := range list {
		if spec, ok := InspectProperty(item); ok {
			out = append(out, spec)
		}
	}
	return out
}

// inspectPropertyMap reads the normalised map spelling of a property. This is
// the shape the runtime's own loader hydrates from, and the shape shared spec
// vectors are written in, so accepting it lets request builders be exercised
// straight from a vector file.
func inspectPropertyMap(m map[string]interface{}) PropertySpec {
	spec := PropertySpec{
		Name:        mapString(m, "name"),
		Kind:        mapString(m, "kind"),
		Description: mapString(m, "description"),
		Required:    mapBool(m, "required"),
		Nullable:    mapBool(m, "nullable"),
	}
	if arr, ok := m["enumValues"].([]interface{}); ok {
		spec.EnumValues = arr
	}
	if items, ok := m["items"]; ok && items != nil {
		if child, ok := InspectProperty(items); ok {
			spec.Items = &child
		}
	}
	spec.Properties = inspectPropertyList(m["properties"])
	spec.OneOf = inspectPropertyList(m["oneOf"])
	spec.AnyOf = inspectPropertyList(m["anyOf"])
	return spec
}

func inspectPropertyList(v interface{}) []PropertySpec {
	list, ok := v.([]interface{})
	if !ok {
		return nil
	}
	return InspectProperties(list)
}

func baseSpec(name, kind string, description *string, required, nullable *bool, enumValues []interface{}) PropertySpec {
	spec := PropertySpec{Name: name, Kind: kind, EnumValues: enumValues}
	if description != nil {
		spec.Description = *description
	}
	if required != nil {
		spec.Required = *required
	}
	if nullable != nil {
		spec.Nullable = *nullable
	}
	return spec
}

func mapString(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func mapBool(m map[string]interface{}, key string) bool {
	if v, ok := m[key].(bool); ok {
		return v
	}
	return false
}
