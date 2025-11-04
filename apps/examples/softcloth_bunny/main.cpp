#include <catch2/catch_all.hpp>
#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/constitution/neo_hookean_shell.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/constitution/discrete_shell_bending.h>
#include <uipc/common/timer.h>
#include <filesystem>
#include <fstream>
#include <numbers>


using Json = nlohmann::json;

template <typename T>
bool convertJsonParameters(const Json& j, T& param)
{
    try
    {
        param = j.get<T>();
    }
    catch(nlohmann::json::exception& e)
    {
        std::cout << e.what() << "\n";
        return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;
    namespace fs = std::filesystem;

    Timer global_timer("main");
    global_timer.enable_all();
    spdlog::set_level(spdlog::level::info);

    // workspace / paths (same pattern as wrecking ball demo)
    std::string tetmesh_dir{AssetDir::tetmesh_path()};
    std::string trimesh_path{AssetDir::trimesh_path()};

    auto        this_output_path = AssetDir::output_path(__FILE__);
    auto        this_folder      = AssetDir::folder(__FILE__);

    // Create engine & world
    Engine engine{"cuda", this_output_path};
    World  world{engine};

    // Scene config (mirror Python settings)
    Json config = Scene::default_config();
    // set parameters similar to Python demo
    config["dt"]                      = 0.01;
    config["contact"]["d_hat"]        = 0.01;
    //config["contact"]["friction"]["enable"] = true;
    //config["contact"]["enable"]             = true;
    //config["contact"]["d_hat"]              = 0.01;
    //config["line_search"]["max_iter"]       = 8;


    // dump config for record
    {
        std::ofstream ofs(fmt::format("{}config.json", this_output_path));
        ofs << config.dump(4);
    }

    Scene scene{config};
    /////////////################################################
    Json wrecking_ball_scene;
    {
        std::ifstream ifs(fmt::format("{}wrecking_ball.json", this_folder));
        ifs >> wrecking_ball_scene;
    }

    Json wrecking_ball_vertexColoring;
    {
        std::ifstream ifs(fmt::format("{}model.VertexColoring.json", this_folder));
        ifs >> wrecking_ball_vertexColoring;
    }

    Json wrecking_ball_edgeColoring;
    {
        std::ifstream ifs(fmt::format("{}model.EdgeColoring.json", this_folder));
        ifs >> wrecking_ball_edgeColoring;
    }

    Json wrecking_ball_tetColoring;
    //{
    //    std::ifstream ifs(fmt::format("{}model.TetColoring.json", this_folder));
    //    ifs >> wrecking_ball_tetColoring;
    //}

    std::vector<std::vector<int32_t>> verticesColoringCategories;
    std::vector<std::vector<int32_t>> edgesColoringCategories;
    std::vector<std::vector<int32_t>> tetsColoringCategories;
    convertJsonParameters(wrecking_ball_vertexColoring, verticesColoringCategories);
    convertJsonParameters(wrecking_ball_edgeColoring, edgesColoringCategories);
    convertJsonParameters(wrecking_ball_tetColoring, tetsColoringCategories);
    // we sort them from large to small for the aggregated solve
    std::sort(tetsColoringCategories.begin(),
              tetsColoringCategories.end(),
              [](const std::vector<int>& a, const std::vector<int>& b)
              { return a.size() > b.size(); });
    // we sort them from large to small for the aggregated solve
    std::sort(tetsColoringCategories.begin(),
              tetsColoringCategories.end(),
              [](const std::vector<int>& a, const std::vector<int>& b)
              { return a.size() > b.size(); });
    /////////////################################################
    S<Object> cloth_obj = scene.objects().create("cloth");
    Float     scale = 2;
    Transform T     = Transform::Identity();
    T.scale(scale);
    SimplicialComplexIO io{T};
    //auto cloth_mesh = io.read(fmt::format("{}grid20x20.obj", trimesh_path));
    auto cloth_mesh = io.read(fmt::format("{}grid10x10.obj", trimesh_path));
    label_surface(cloth_mesh);
    NeoHookeanShell nhs;
    scene.constitution_tabular().insert(nhs);


    DiscreteShellBending dsb;

    auto parm = ElasticModuli::youngs_poisson(100.0_kPa, 0.499);
    nhs.apply_to(cloth_mesh, parm, 200, 0.001);
    dsb.apply_to(cloth_mesh, 10.0_Pa);

    auto pos_view = view(cloth_mesh.positions());
    //std::ranges::transform(pos_view,
    //                       pos_view.begin(),
    //                       [](const Vector3& v) -> Vector3
    //                       { return v + Vector3{1, 0.6, 1}; });

    std::ranges::transform(pos_view,
                           pos_view.begin(),
                           [](const Vector3& v) -> Vector3
                           { return v + Vector3{1, 1.6, 1}; });

    auto is_fixed      = cloth_mesh.vertices().find<IndexT>(builtin::is_fixed);
    view(*is_fixed)[0] = 1;
    view(*is_fixed)[1] = 1;
    view(*is_fixed)[10]  = 1;
    //view(*is_fixed)[11] = 1;
    //view(*is_fixed)[12]  = 1;
    //view(*is_fixed)[22] = 1;

    auto is_self_collision = cloth_mesh.meta().find<IndexT>(builtin::self_collision);
    //auto is_self_collision = cloth_mesh.vertices().find<IndexT>(builtin::self_collision);
    //std::ranges::fill(view(*is_self_collision), 0);


    cloth_obj->geometries().create(cloth_mesh);

    S<Object> bunny_obj = scene.objects().create("bunny");
    Transform T1         = Transform::Identity();
    T1.translate(Vector3::UnitX() + Vector3::UnitZ());
    SimplicialComplexIO io1{T1};
    //auto bunny_mesh = io1.read(fmt::format("{}bunny0.msh", tetmesh_dir));
    auto bunny_mesh = io1.read(fmt::format("{}cube.msh", tetmesh_dir));
    label_surface(bunny_mesh);
    label_triangle_orient(bunny_mesh);

    int cloth_vertex_size = cloth_mesh.vertices().size();
    int bunny_vertex_size=bunny_mesh.vertices().size();

    //auto default_contact = scene.contact_tabular().default_element();
    //default_contact.apply_to(cloth_mesh);
    //default_contact.apply_to(bunny_mesh);

    auto flipped_bunny_mesh = flip_inward_triangles(bunny_mesh);
    // create constitution and contact as in Python


    //StableNeoHookean sth;
    //scene.constitution_tabular().insert(nhs);
    //auto parm1 = ElasticModuli::youngs_poisson(100.0_MPa, 0.499);
    //sth.apply_to(flipped_bunny_mesh, parm1);
    //auto is_fixed = flipped_bunny_mesh.instances().find<IndexT>(builtin::is_fixed);

    //AffineBodyConstitution abd;
    //scene.constitution_tabular().insert(abd);
    //abd.apply_to(flipped_bunny_mesh, 100.0_MPa);
    //auto is_fixed = flipped_bunny_mesh.instances().find<IndexT>(builtin::is_fixed);
    //view(*is_fixed)[0] = 1;
    //bunny_obj->geometries().create(flipped_bunny_mesh);
    
    // create ground (as in Python)
    Float ground_height = -1.0;
    auto  g             = geometry::ground(ground_height);
    auto  ground_obj    = scene.objects().create("ground");
    //ground_obj->geometries().create(g);

    // init world & scene IO
    world.init(scene);
    SceneIO sio{scene};
    sio.write_surface(fmt::format("{}scene_surface{}.obj", this_output_path, 0));

    // optional: get some geometry handle (example from wrecking ball)
    // advance simulation for N frames and dump surfaces (no GUI)
    const int MaxFrames = 200;
    while(world.frame() < MaxFrames)
    {
        world.advance();
        world.retrieve();

        // write surface each frame (like wrecking ball)
        sio.write_surface(
            fmt::format("{}scene_surface{}.obj", this_output_path, world.frame()));

        // (optional) log progress every 50 frames
        if(world.frame() % 50 == 0)
            spdlog::info("Frame {}", world.frame());
    }

    // final timing report
    global_timer.report();

    return 0;
}