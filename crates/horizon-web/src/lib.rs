//! Horizon WebAssembly browser client entrypoint.
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

#[derive(Default)]
struct HorizonWebApp {
    canvas_id: String,
    window: Option<Arc<Window>>,
}

impl HorizonWebApp {
    fn new(canvas_id: &str) -> Self {
        Self {
            canvas_id: canvas_id.to_owned(),
            window: None,
        }
    }
}

impl ApplicationHandler for HorizonWebApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        use winit::platform::web::WindowAttributesExtWebSys;

        let window_target = web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.get_element_by_id(&self.canvas_id));

        let mut attributes = Window::default_attributes().with_title("Horizon Web");

        if let Some(canvas) = window_target.and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok()) {
            attributes = attributes.with_canvas(Some(canvas));
        } else {
            attributes = attributes.with_append(true);
        }

        match event_loop.create_window(attributes) {
            Ok(win) => {
                let win = Arc::new(win);
                self.window = Some(win.clone());
                log::info!("Horizon Web window created successfully.");
            }
            Err(e) => {
                log::error!("Failed to create Web window: {e}");
            }
        }
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::RedrawRequested => {
                // Main render frame tick
            }
            WindowEvent::Resized(size) => {
                log::debug!("Window resized to: {}x{}", size.width, size.height);
            }
            _ => {}
        }
    }
}

/// Browser initialization entrypoint called from JavaScript module.
#[wasm_bindgen]
pub fn run_horizon(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);

    log::info!("Horizon Minecraft WebAssembly engine initializing on canvas '#{}'...", canvas_id);

    let event_loop = EventLoop::new().map_err(|e| JsValue::from_str(&e.to_string()))?;

    use winit::platform::web::EventLoopExtWebSys;
    let app = HorizonWebApp::new(canvas_id);
    event_loop.spawn_app(app);

    Ok(())
}
