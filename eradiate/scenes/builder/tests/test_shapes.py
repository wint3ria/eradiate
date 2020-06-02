from eradiate.scenes.builder.bsdfs import Diffuse
from eradiate.scenes.builder.media import Homogeneous
from eradiate.scenes.builder.shapes import Cube, Rectangle, Ply
from eradiate.scenes.builder.spectra import Spectrum
from eradiate.scenes.builder.transforms import Scale, Transform, Translate


def test_rectangle(variant_scalar_mono):
    # Default init
    r = Rectangle()
    assert r.to_xml() == """<shape type="rectangle"/>"""
    r.instantiate()

    # Init with params
    r = Rectangle(bsdf=Diffuse(), to_world=Transform([Scale(2.)]))
    assert r.to_xml() == \
        '<shape type="rectangle">' \
        '<transform name="to_world"><scale value="2.0"/></transform>' \
        '<bsdf type="diffuse"/>' \
        '</shape>'
    r.instantiate()


def test_cube():
    # Default init
    c = Cube()
    assert c.to_xml() == """<shape type="cube"/>"""
    c.instantiate()

    # Init with params
    c = Cube(
        interior=Homogeneous(sigma_t=Spectrum(1.0),
        albedo=Spectrum(0.5)),
        to_world=Transform([Translate([0., 0., 50.])])
    )
    assert c.to_xml() == \
        '<shape type="cube">' \
        '<transform name="to_world">' \
        '<translate value="0.0, 0.0, 50.0"/>' \
        '</transform>' \
        '<medium name="interior" type="homogeneous">' \
        '<spectrum name="sigma_t" value="1.0"/>' \
        '<spectrum name="albedo" value="0.5"/>' \
        '</medium>' \
        '</shape>'
    c.instantiate()

def test_ply():
    # Default init
    fname = "ext/mitsuba2/resources/data/ply/sphere.ply"
    p = Ply(filename=fname)
    assert p.to_xml() == f"""<shape type="ply"><string name="filename" value="{fname}"/></shape>"""
    p.instantiate()

    # Init with params
    p = Ply(
        filename=fname,
        bsdf=Diffuse(),
        interior=Homogeneous(sigma_t=Spectrum(1.0),
        albedo=Spectrum(0.5)),
        to_world=Transform([Translate([0., 0., 50.])])
    )
    assert p.to_xml() == \
        '<shape type="ply">' \
        '<transform name="to_world">' \
        '<translate value="0.0, 0.0, 50.0"/>' \
        '</transform>' \
        '<medium name="interior" type="homogeneous">' \
        '<spectrum name="sigma_t" value="1.0"/>' \
        '<spectrum name="albedo" value="0.5"/>' \
        '</medium>' \
        '<string name="filename" value="ext/mitsuba2/resources/data/ply/sphere.ply"/>' \
        '<bsdf type="diffuse"/>' \
        '</shape>'
    p.instantiate()
