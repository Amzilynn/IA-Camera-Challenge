import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import colorsys

class UIProcessor:
    def __init__(self, font_path="C:\\Windows\\Fonts\\arial.ttf"):
        self.font_path = font_path if os.path.exists(font_path) else None
        # Use more diverse font sizes for hierarchy
        self.fonts = {
            'tiny': ImageFont.truetype(self.font_path, 11) if self.font_path else None,
            'small': ImageFont.truetype(self.font_path, 14) if self.font_path else None,
            'medium': ImageFont.truetype(self.font_path, 18) if self.font_path else None,
            'large': ImageFont.truetype(self.font_path, 24) if self.font_path else None,
            'huge': ImageFont.truetype(self.font_path, 36) if self.font_path else None,
            'display': ImageFont.truetype(self.font_path, 48) if self.font_path else None
        }
        
        # Premium Color Palette (Vibrant & Deep)
        self.colors = {
            'staff': (255, 120, 0),        # Vibrant Orange
            'visitor': (0, 180, 255),      # Electric Blue
            'happy': (0, 255, 150),        # Neon Mint
            'neutral': (180, 180, 200),    # Slate Gray
            'sad': (255, 50, 50),          # Bright Red
            'accent': (0, 255, 255),       # Cyan Neon
            'bg_dark': (10, 10, 15, 210),  # Deep glass
            'bg_light': (40, 40, 50, 180), # Header glass
            'text_main': (255, 255, 255),
            'text_dim': (160, 160, 180)
        }

    def _cv2_to_pil(self, img):
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _pil_to_cv2(self, img):
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def draw_glass_panel(self, pil_img, overlay_layer, pos, size, radius=10, blur=15):
        """Draws a real frosted glass panel using blurred background."""
        x, y = pos
        w, h = size
        
        # 1. Crop the background area
        box = (x, y, x + w, y + h)
        if x < 0 or y < 0 or x + w > pil_img.width or y + h > pil_img.height:
            # Fallback for out of bounds
            draw = ImageDraw.Draw(overlay_layer)
            draw.rectangle([x, y, x + w, y + h], fill=self.colors['bg_dark'])
            return

        region = pil_img.crop(box)
        
        # 2. Apply strong Gaussian Blur to the region
        region = region.filter(ImageFilter.GaussianBlur(radius=blur))
        
        # 3. Paste blurred region back to a temporary layer to avoid recursive blur
        # But since we are modifying the PIL image directly, we have to be careful.
        # However, for HUD it's usually fine.
        pil_img.paste(region, box)
        
        # 4. Overlay color and stroke for "glass" look on the overlay layer
        draw_overlay = ImageDraw.Draw(overlay_layer)
        draw_overlay.rectangle([x, y, x+w, y+h], fill=self.colors['bg_dark'])
        
        # Subtle border
        draw_overlay.rectangle([x, y, x+w, y+h], outline=(255, 255, 255, 40), width=1)

    def draw_gradient_bar(self, draw, pos, size, progress, color_start, color_end):
        """Draws a smooth gradient progress bar."""
        x, y = pos
        w, h = size
        
        # Background track
        draw.rectangle([x, y, x + w, y + h], fill=(30, 30, 40, 255))
        
        # Fill bar
        fill_w = int(w * (progress / 100))
        if fill_w > 1:
            for i in range(fill_w):
                # Interpolate color
                ratio = i / fill_w
                r = int(color_start[0] * (1 - ratio) + color_end[0] * ratio)
                g = int(color_start[1] * (1 - ratio) + color_end[1] * ratio)
                b = int(color_start[2] * (1 - ratio) + color_end[2] * ratio)
                draw.line([(x + i, y), (x + i, y + h)], fill=(r, g, b, 255))
            
            # Add a small glow at the tip
            tip_x = x + fill_w
            draw.line([(tip_x, y), (tip_x, y + h)], fill=(255, 255, 255, 200), width=2)
        elif fill_w == 1:
            draw.line([(x, y), (x, y + h)], fill=color_start, width=1)

    def draw_scanlines(self, draw, size):
        """Draws a subtle scanline overlay for the cinematic HUD feel."""
        w, h = size
        for y in range(0, h, 4):
            draw.line([(0, y), (w, y)], fill=(255, 255, 255, 15), width=1)

    def draw_status_icon(self, draw, pos, icon_type, color):
        """Draws a stylized icon (Person vs Staff) on the person card."""
        x, y = pos
        if icon_type == "staff":
            # Stylized badge icon
            draw.polygon([(x, y+2), (x+8, y), (x+16, y+2), (x+16, y+10), (x+8, y+14), (x, y+10)], fill=color)
        else:
            # Stylized person circle
            draw.ellipse([x+4, y, x+12, y+8], fill=color)
            draw.arc([x, y+6, x+16, y+16], start=180, end=0, fill=color, width=2)

    def render_hud(self, frame, detections, interactions, global_metrics):
        """Ultra-Premium Cyber-Agent HUD with Relational De-cluttering."""
        orig_h, orig_w = frame.shape[:2]
        pil_img = self._cv2_to_pil(frame).convert('RGBA')
        
        overlay_layer = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
        scanline_layer = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_layer)
        draw_scan = ImageDraw.Draw(scanline_layer)

        # 0. GLOBAL AMBIENT GLOW
        idx = global_metrics.get('satisfaction_index', 0)
        vibe_color = self.colors['happy'] if idx > 75 else self.colors['sad'] if idx < 50 else self.colors['accent']
        
        # Edge Glow
        for i in range(30):
            alpha = int(30 * (1 - i/30))
            draw.rectangle([i, i, orig_w-i, orig_h-i], outline=(*vibe_color, alpha), width=1)

        # 1. RELATIONSHIP DE-CLUTTERING & RENDERING
        # Deduplication: One interaction per pair, based on priority
        priority = {"Service/Helping": 6, "Talking": 5, "Physical Contact": 4, "Approaching": 3, "Walking Together": 2, "Group_Bond": 1}
        best_interactions = {}
        for inter in interactions:
            pair = tuple(sorted(inter['ids']))
            itype = inter['type']
            if pair not in best_interactions or priority.get(itype, 0) > priority.get(best_interactions[pair]['type'], 0):
                best_interactions[pair] = inter
        
        for inter in best_interactions.values():
            try:
                id1, id2 = inter['ids']
                itype = inter['type']
                p1 = next((d for d in detections if d.get('track_id_original', d.get('track_id')) == id1), None)
                p2 = next((d for d in detections if d.get('track_id_original', d.get('track_id')) == id2), None)
                if not p1 or not p2: continue
                
                c1 = ((p1['bbox'][0] + p1['bbox'][2]) / 2, (p1['bbox'][1] + p1['bbox'][3]) / 2)
                c2 = ((p2['bbox'][0] + p2['bbox'][2]) / 2, (p2['bbox'][1] + p2['bbox'][3]) / 2)
                
                h_avg = ((p1['bbox'][3]-p1['bbox'][1]) + (p2['bbox'][3]-p2['bbox'][1]))/2
                if np.linalg.norm(np.array(c1) - np.array(c2)) > h_avg * 3.5: continue

                line_color = self.colors['happy'] if itype == "Talking" else self.colors['accent']
                draw.line([c1, c2], fill=(*line_color, 180), width=1)
                
                # Floating Label
                mid = ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
                t_label = itype.upper()
                tw = draw.textlength(t_label, font=self.fonts['tiny'])
                draw.rectangle([mid[0]-tw/2-6, mid[1]-9, mid[0]+tw/2+6, mid[1]+9], fill=(10, 10, 20, 240), outline=(*line_color, 120))
                draw.text((mid[0]-tw/2, mid[1]-7), t_label, font=self.fonts['tiny'], fill=(255,255,255))
            except: continue

        # 2. PERSON CARDS
        for det in detections:
            if det['type'] != 'person': continue
            x1, y1, x2, y2 = det['bbox']
            role = det.get('role', 'Analyzing')
            emotion = det.get('emotion', 'Neutral').capitalize()
            color = self.colors['staff'] if "Staff" in role else self.colors['visitor']
            
            # Crosshair Corners
            l = 10
            draw.line([(x1, y1), (x1+l, y1)], fill=color, width=2); draw.line([(x1, y1), (x1, y1+l)], fill=color, width=2)
            draw.line([(x2, y1), (x2-l, y1)], fill=color, width=2); draw.line([(x2, y1), (x2, y1+l)], fill=color, width=2)
            draw.line([(x1, y2), (x1+l, y2)], fill=color, width=2); draw.line([(x1, y2), (x1, y2-l)], fill=color, width=2)
            draw.line([(x2, y2), (x2-l, y2)], fill=color, width=2); draw.line([(x2, y2), (x2, y2-l)], fill=color, width=2)
            
            cw, ch = 140, 38
            cx, cy = x1, y1 - ch - 10
            if cy < 0: cy = y2 + 10
            
            self.draw_glass_panel(pil_img, overlay_layer, (int(cx), int(cy)), (cw, ch), blur=10)
            draw.rectangle([cx, cy, cx+3, cy+ch], fill=color)
            self.draw_status_icon(draw, (cx + 8, cy + 5), "staff" if "Staff" in role else "visitor", color)
            draw.text((cx + 28, cy + 4), role.upper(), font=self.fonts['tiny'], fill=self.colors['text_dim'])
            draw.text((cx + 10, cy + 18), f"{emotion.upper()}", font=self.fonts['small'], fill=self.colors['text_main'])

        # 3. PREMIUM DASHBOARD (Top Right)
        margin = 30
        dash_w, dash_h = 320, 180
        dash_x = orig_w - dash_w - margin
        dash_y = margin
        
        # Dashboard Glass Panel
        self.draw_glass_panel(pil_img, overlay_layer, (dash_x, dash_y), (dash_w, dash_h), radius=20, blur=25)
        
        # Title
        title_font = self.fonts['medium']
        draw.text((dash_x + 20, dash_y + 15), "SCENE INTELLIGENCE", font=title_font, fill=(200, 0, 200, 255)) # Pinkish magenta
        
        # Stats
        stats_labels = [
            ("TOTAL PEOPLE", f"{global_metrics.get('total_people', 0)}"),
            ("STAFF / VISITORS", f"{global_metrics.get('staff_count', 0)} / {global_metrics.get('visitor_count', 0)}"),
            ("ACTIVE INTERACT", f"{global_metrics.get('active_engagements', 0)}"),
            ("SATISFACTION", f"{int(idx)}%")
        ]
        
        stats_font = self.fonts['small']
        val_font = self.fonts['medium']
        
        for i, (label, val) in enumerate(stats_labels):
            y_off = dash_y + 55 + (i * 30)
            draw.text((dash_x + 20, y_off), label, font=stats_font, fill=self.colors['text_dim'])
            
            # Right-aligned value
            text_w = draw.textlength(val, font=val_font)
            draw.text((dash_x + dash_w - text_w - 20, y_off - 4), val, font=val_font, fill=self.colors['text_main'])

        # Final Compositing
        self.draw_scanlines(draw_scan, (orig_w, orig_h))
        scanline_overlay = Image.alpha_composite(overlay_layer, scanline_layer)
        combined = Image.alpha_composite(pil_img, scanline_overlay)
        return self._pil_to_cv2(combined.convert('RGB'))
